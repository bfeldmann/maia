<?php

namespace Biigle\Modules\Maia\Jobs;

use Biigle\Modules\Maia\GenericImage;
use Biigle\Modules\Maia\MaiaJob;
use Biigle\Volume;
use Exception;
use File;
use FileCache;

class InstanceSegmentationRequest extends JobRequest
{
    /**
     * Selected training proposals.
     *
     * @var array
     */
    protected $trainingProposals;

    /**
     * URL of the volume for knowledge transfer (if any).
     *
     * @var string
     */
    protected $knowledgeTransferVolumeUrl;

    /**
     * Filenames of the images of the knowledge transfer volume, indexed by their IDs.
     *
     * @var array
     */
    protected $knowledgeTransferImages;

    /**
     * Create a new instance
     *
     * @param MaiaJob $job
     */
    public function __construct(MaiaJob $job)
    {
        parent::__construct($job);
        // Make sure to convert the annotations to arrays because it is more efficient
        // and the GPU server cannot instantiate MaiaAnnotation objects (as they depend
        // on biigle/core).
        $this->trainingProposals = $this->bundleTrainingProposals($job);

        if ($this->shouldUseKnowledgeTransfer()) {
            $volume = Volume::find($this->jobParams['kt_volume_id']);
            $this->knowledgeTransferVolumeUrl = $volume->url;
            $this->knowledgeTransferImages = $volume->images()
                ->pluck('filename', 'id')
                ->toArray();
        }
    }

    /**
     * Execute the job
     */
    public function handle()
    {
        $this->createTmpDir();

        try {
            $images = $this->getGenericImages();

            if ($this->shouldUseKnowledgeTransfer()) {
                $datasetImages = $this->getKnowledgeTransferImages();
            } else {
                $datasetImages = $images;
            }

            $datasetOutputPath = $this->generateDataset($datasetImages);
            $trainingOutputPath = $this->performTraining($datasetOutputPath);
            $this->performInference($images, $datasetOutputPath, $trainingOutputPath);

            $annotations = $this->parseAnnotations($images);

            # ---> TODO: Check if working correctly, write test
            $crops = $this->cropImages($images, $datasetOutputPath, $annotations);
            $fvs = $this->generateDINOFV($crops);
            # <---

            $this->dispatchResponse($annotations);
        } finally {
            $this->cleanup();
        }
    }

    /**
     * TODO: Test usage
     *
     * @param array $imagesMap Map from image IDs to cached file paths.
     * @param string $outputJsonPath Path to the output file of the script.
     * @param array $annotations Array of the parsed annotations.
     *
     * @return string Input JSON file path.
     */
    protected function createCropImagesJson($imagesMap, $outputJsonPath, $annotations)
    {
        $path = "{$this->tmpDir}/input-crop_images.json";
        $content = [
            'images' => $imagesMap,
            'annotations' => $annotations,
            'tmp_dir' => $this->tmpDir,
            'max_workers' => intval(config('maia.max_workers')),
            'output_path' => $outputJsonPath,
            'resize_dimension' => intval(config('maia.resize_dimension')),
        ];

        File::put($path, json_encode($content, JSON_UNESCAPED_SLASHES));

        return $path;
    }

    /**
     * TODO: Test usage
     *
     * @param array $images GenericImage instances.
     * @param string $datasetOutputPath Path to the JSON output of the dataset generator.
     * @param array $annotations Array of the parsed annotations.
     *
     * @return string Path to the JSON output file of the cropping script.
     */
    protected function cropImages($images, $datasetOutputPath, $annotations)
    {
        $outputPath = "{$this->tmpDir}/output-cropping.json";

        FileCache::batch($relevantImages, function ($images, $paths) use ($datasetOutputPath) {
            $imagesMap = $this->buildImagesMap($images, $paths);
            $inputPath = $this->createCropJson($imagesMap, $outputPath, $annotations);
            $script = config('maia.crop_images_script');
            $this->python("{$script} {$inputPath}", 'cropimages-log.txt');
        });

        return $outputPath;
    }

    /**
     * TODO: Test usage
     *
     * @param string $cropImagesJson Path to the output file of the image cropping.
     * @param string $outputJsonPath Path to the output file of the script.
     *
     * @return string Input JSON file path.
     */
    protected function createDINOJson($cropImagesJson, $outputJsonPath, $backbone)
    {
        $cropJson = json_decode($cropImagesJson);

        $path = "{$this->tmpDir}/input-dinofv.json";
        $content = [
            'cropped_images_path' => $cropJson->cropped_images_path,
            'cropped_images' => $cropJson->cropped_images,
            'resize_dimension' => $cropJson->resize_dimension,
            'backbone' => $backbone,
            'tmp_dir' => $this->tmpDir,
            'max_workers' => intval(config('maia.max_workers')),
            'output_path' => $outputJsonPath,
        ];

        File::put($path, json_encode($content, JSON_UNESCAPED_SLASHES));

        return $path;
    }

    /**
     * TODO: Write tests, test php and python functions.
     * Perform DINO feature vector generation.
     *
     * @param string $cropOutputPath Path to the JSON output of the image cropping.
     *
     * @return string Path to the JSON output file of the feature vector generation script.
     */
    protected function generateDINOFV($cropOutputPath)
    {
        $outputPath = "{$this->tmpDir}/output-dinofv.json";
        $backbone = config('maia.dino_model');
        $inputPath = $this->createDINOJson($cropOutputPath, $outputPath, $backbone);
        $script = config('maia.dino_fv_script');

        $this->python("{$script} {$inputPath}", 'dinofv-log.txt');

        return $outputPath;
    }

    /**
     * Determine whether knowledge transfer should be performed in this job.
     *
     * @return bool
     */
    protected function shouldUseKnowledgeTransfer()
    {
        return array_key_exists('training_data_method', $this->jobParams) && in_array($this->jobParams['training_data_method'], ['knowledge_transfer', 'area_knowledge_transfer']);
    }

    /**
     * Bundle the training proposals to be sent to the GPU server.
     *
     * @param MaiaJob $job
     *
     * @return array
     */
    protected function bundleTrainingProposals(MaiaJob $job)
    {
        return $job->trainingProposals()
            ->selected()
            ->select('image_id', 'points')
            ->get()
            ->groupBy('image_id')
            ->map(function ($proposals) {
                return $proposals->pluck('points')->map(function ($proposal) {
                    // The circles of the proposals are drawn by OpenCV and this expects
                    // integers. As we can shave off a few bytes of job payload this
                    // way, we parse the coordinates here instead of in the Python
                    // script.
                    return array_map(function ($value) {
                        return intval(round($value));
                    }, $proposal);
                });
            })
            ->toArray();
    }

    /**
     * Generate the training dataset for Mask R-CNN.
     *
     * @param array $images GenericImage instances.
     *
     * @return string Path to the JSON output file.
     */
    protected function generateDataset($images)
    {
        $outputPath = "{$this->tmpDir}/output-dataset.json";

        // All images that contain selected training proposals.
        $relevantImages = array_filter($images, function ($image) {
            return array_key_exists($image->getId(), $this->trainingProposals);
        });

        FileCache::batch($relevantImages, function ($images, $paths) use ($outputPath) {
            $imagesMap = $this->buildImagesMap($images, $paths);
            $inputPath = $this->createDatasetJson($imagesMap, $outputPath);
            $script = config('maia.mrcnn_dataset_script');
            $this->python("{$script} {$inputPath}", 'dataset-log.txt');
        });

        return $outputPath;
    }

    /**
     * Create the JSON file that is the input to the dataset generation script.
     *
     * @param array $imagesMap Map from image IDs to cached file paths.
     * @param string $outputJsonPath Path to the output file of the script.
     *
     * @return string Input JSON file path.
     */
    protected function createDatasetJson($imagesMap, $outputJsonPath)
    {
        $path = "{$this->tmpDir}/input-dataset.json";
        $content = [
            'images' => $imagesMap,
            'tmp_dir' => $this->tmpDir,
            'available_bytes' => intval(config('maia.available_bytes')),
            'max_workers' => intval(config('maia.max_workers')),
            'training_proposals' => $this->trainingProposals,
            'output_path' => $outputJsonPath,
        ];

        if ($this->shouldUseKnowledgeTransfer()) {
            $content['kt_scale_factors'] = $this->jobParams['kt_scale_factors'];
        }

        File::put($path, json_encode($content, JSON_UNESCAPED_SLASHES));

        return $path;
    }

    /**
     * Perform training of Mask R-CNN.
     *
     * @param string $datasetOutputPath Path to the JSON output of the dataset generator.
     *
     * @return string Path to the JSON output file of the training script.
     */
    protected function performTraining($datasetOutputPath)
    {
        $outputPath = "{$this->tmpDir}/output-training.json";
        $this->maybeDownloadCocoModel();
        $inputPath = $this->createTrainingJson($outputPath);
        $script = config('maia.mrcnn_training_script');
        $this->python("{$script} {$inputPath} {$datasetOutputPath}", 'training-log.txt');

        return $outputPath;
    }

    /**
     * Downloads the Mask R-CNN COCO pretrained weights if they weren't downloaded yet.
     */
    protected function maybeDownloadCocoModel()
    {
        $path = config('maia.coco_model_path');
        if (!File::exists($path)) {
            $this->ensureDirectory(dirname($path));
            $url = config('maia.coco_model_url');
            $success = @copy($url, $path);

            if (!$success) {
                throw new Exception("Failed to download Mask R-CNN weights from '{$url}'.");
            }
        }
    }

    /**
     * Create the JSON file that is the input to the training script.
     *
     * @param string $outputJsonPath Path to the output file of the script.
     *
     * @return string Input JSON file path.
     */
    protected function createTrainingJson($outputJsonPath)
    {
        $path = "{$this->tmpDir}/input-training.json";
        $content = [
            'is_train_scheme' => $this->jobParams['is_train_scheme'],
            'tmp_dir' => $this->tmpDir,
            'available_bytes' => intval(config('maia.available_bytes')),
            'max_workers' => intval(config('maia.max_workers')),
            'output_path' => $outputJsonPath,
            'coco_model_path' => config('maia.coco_model_path'),
        ];

        File::put($path, json_encode($content, JSON_UNESCAPED_SLASHES));

        return $path;
    }

    /**
     * Perform inference with the trained Mask R-CNN.
     *
     * @param array $images GenericImage instances.
     * @param string $datasetOutputPath Path to the JSON output of the dataset generator.
     * @param string $trainingOutputPath Path to the JSON output of the training script.
     */
    protected function performInference($images, $datasetOutputPath, $trainingOutputPath)
    {
        FileCache::batch($images, function ($images, $paths) use ($datasetOutputPath, $trainingOutputPath) {
            $imagesMap = $this->buildImagesMap($images, $paths);
            $inputPath = $this->createInferenceJson($imagesMap);
            $script = config('maia.mrcnn_inference_script');
            $this->python("{$script} {$inputPath} {$datasetOutputPath} {$trainingOutputPath}", 'inference-log.txt');
        });
    }

    /**
     * Create the JSON file that is the input to the inference script.
     *
     * @param array $imagesMap Map from image IDs to cached file paths.
     *
     * @return string Input JSON file path.
     */
    protected function createInferenceJson($imagesMap)
    {
        $path = "{$this->tmpDir}/input-inference.json";
        $content = [
            'images' => $imagesMap,
            'tmp_dir' => $this->tmpDir,
            'available_bytes' => intval(config('maia.available_bytes')),
            'max_workers' => intval(config('maia.max_workers')),
        ];

        File::put($path, json_encode($content, JSON_UNESCAPED_SLASHES));

        return $path;
    }

    /**
     * Build the map from image ID to path of the cached image file.
     *
     * @param array $images GenericImage instances.
     * @param array $paths Cached image file paths.
     *
     * @return array
     */
    protected function buildImagesMap($images, $paths)
    {
        $imagesMap = [];
        foreach ($images as $index => $image) {
            $imagesMap[$image->getId()] = $paths[$index];
        }

        return $imagesMap;
    }

    /**
     * Dispatch the job to store the instance segmentation results.
     *
     * @param array $annotations
     */
    protected function dispatchResponse($annotations)
    {
        $this->dispatch(new InstanceSegmentationResponse($this->jobId, $annotations));
    }

    /**
     * {@inheritdoc}
     */
    protected function dispatchFailure(Exception $e)
    {
        $this->dispatch(new InstanceSegmentationFailure($this->jobId, $e));
    }

    /**
     * {@inheritdoc}
     */
    protected function getTmpDirPath()
    {
        return parent::getTmpDirPath()."-instance-segmentation";
    }

    /**
     * Create GenericImage instances for the images of the knowledge transfer volume.
     *
     * @return array
     */
    protected function getKnowledgeTransferImages()
    {
        $images = [];
        foreach ($this->knowledgeTransferImages as $id => $filename) {
            $images[$id] = new GenericImage($id, "{$this->knowledgeTransferVolumeUrl}/{$filename}");
        }

        return $images;
    }
}
