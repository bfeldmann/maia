<?php

namespace Biigle\Tests\Modules\Maia\Http\Controllers\Api;

use ApiTestCase;
use Biigle\MediaType;
use Biigle\Modules\Maia\Jobs\InstanceSegmentationRequest;
use Biigle\Modules\Maia\MaiaJob;
use Biigle\Modules\Maia\MaiaJobState as State;
use Biigle\Tests\ImageAnnotationLabelTest;
use Biigle\Tests\ImageAnnotationTest;
use Biigle\Tests\ImageTest;
use Biigle\Tests\Modules\Maia\MaiaJobTest;
use Biigle\Tests\Modules\Maia\TrainingProposalTest;

class MaiaJobControllerTest extends ApiTestCase
{
    protected $defaultParams;

    public function setUp(): void
    {
        parent::setUp();
        $this->defaultParams = [
            'training_data_method' => 'novelty_detection',
            'nd_clusters' => 1,
            'nd_patch_size' => 39,
            'nd_threshold' => 99,
            'nd_latent_size' => 0.1,
            'nd_trainset_size' => 10000,
            'nd_epochs' => 100,
            'nd_stride' => 2,
            'nd_ignore_radius' => 5,
            'is_train_scheme' => [
                ['layers' => 'heads', 'epochs' => 10, 'learning_rate' => 0.001],
                ['layers' => 'all', 'epochs' => 10, 'learning_rate' => 0.0001],
            ],
        ];
        ImageTest::create(['volume_id' => $this->volume()->id]);
    }

    public function testStoreNoveltyDetection()
    {
        $id = $this->volume()->id;
        $this->doTestApiRoute('POST', "/api/v1/volumes/{$id}/maia-jobs");

        $this->beGuest();
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs")->assertStatus(403);

        $this->beEditor();
        // mssing arguments
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs")->assertStatus(422);

        // patch size must be an odd number
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", [
            'training_data_method' => 'novelty_detection',
            'nd_clusters' => 5,
            'nd_patch_size' => 40,
            'nd_threshold' => 99,
            'nd_latent_size' => 0.1,
            'nd_trainset_size' => 10000,
            'nd_epochs' => 100,
            'nd_stride' => 2,
            'nd_ignore_radius' => 5,
            'is_train_scheme' => [
                ['layers' => 'heads', 'epochs' => 10, 'learning_rate' => 0.001],
                ['layers' => 'all', 'epochs' => 10, 'learning_rate' => 0.0001],
            ],
        ])->assertStatus(422);

        // empty train scheme
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", [
            'training_data_method' => 'novelty_detection',
            'nd_clusters' => 5,
            'nd_patch_size' => 40,
            'nd_threshold' => 99,
            'nd_latent_size' => 0.1,
            'nd_trainset_size' => 10000,
            'nd_epochs' => 100,
            'nd_stride' => 2,
            'nd_ignore_radius' => 5,
            'is_train_scheme' => [],
        ])->assertStatus(422);

        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $this->defaultParams)
            ->assertSuccessful();

        $job = MaiaJob::first();
        $this->assertNotNull($job);
        $this->assertEquals($id, $job->volume_id);
        $this->assertEquals($this->editor()->id, $job->user_id);
        $this->assertEquals(State::noveltyDetectionId(), $job->state_id);
        $this->assertEquals($this->defaultParams, $job->params);

        // only one running job at a time
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $this->defaultParams)
            ->assertStatus(422);
    }

    public function testStoreFailedNoveltyDetection()
    {
        $id = $this->volume()->id;
        $job = MaiaJobTest::create([
            'state_id' => State::failedNoveltyDetectionId(),
            'volume_id' => $this->volume()->id,
        ]);

        $this->beEditor();
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $this->defaultParams)
            ->assertSuccessful();
    }

    public function testStoreFailedInstanceSegmentation()
    {
        $id = $this->volume()->id;
        $job = MaiaJobTest::create([
            'state_id' => State::failedInstanceSegmentationId(),
            'volume_id' => $this->volume()->id,
        ]);

        $this->beEditor();
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $this->defaultParams)
            ->assertSuccessful();
    }

    public function testStoreTiledImages()
    {
        $id = $this->volume()->id;
        ImageTest::create(['volume_id' => $id, 'tiled' => true, 'filename' => 'x']);

        $this->beEditor();
        // MAIA is not available for volumes with tiled images.
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $this->defaultParams)
            ->assertStatus(422);
    }

    public function testStoreVideoVolume()
    {
        $volume = $this->volume();
        $volume->media_type_id = MediaType::videoId();
        $volume->save();

        $this->beEditor();
        $this->postJson("/api/v1/volumes/{$volume->id}/maia-jobs", $this->defaultParams)
            ->assertStatus(422);
    }

    public function testStoreUseExistingAnnotations()
    {
        $id = $this->volume()->id;
        $this->beEditor();
        $params = [
            'training_data_method' => 'own_annotations',
            'is_train_scheme' => [
                ['layers' => 'heads', 'epochs' => 10, 'learning_rate' => 0.001],
                ['layers' => 'all', 'epochs' => 10, 'learning_rate' => 0.0001],
            ],
        ];
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $params)
            // No existing annotations.
            ->assertStatus(422);

        ImageAnnotationTest::create([
            'image_id' => ImageTest::create([
                'volume_id' => $this->volume()->id,
                'filename' => 'abc.jpg',
            ])->id,
        ]);

        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $params)
            ->assertSuccessful();
        $job = MaiaJob::first();
        $this->assertTrue($job->shouldUseExistingAnnotations());
        $this->assertEquals(State::instanceSegmentationId(), $job->state_id);
    }

    public function testStoreRestrictLabels()
    {
        $id = $this->volume()->id;
        $this->beEditor();
        $params = [
            'training_data_method' => 'novelty_detection',
            'oa_restrict_labels' => [$this->labelChild()->id],
            'is_train_scheme' => [
                ['layers' => 'heads', 'epochs' => 10, 'learning_rate' => 0.001],
                ['layers' => 'all', 'epochs' => 10, 'learning_rate' => 0.0001],
            ],
        ];

        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $params)
            // Requires 'own_annotations'.
            ->assertStatus(422);

        $params['training_data_method'] = 'own_annotations';
        $params['oa_restrict_labels'] = [999];
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $params)
            // Must contain valid label IDs.
            ->assertStatus(422);

        ImageAnnotationLabelTest::create([
            'label_id' => $this->labelChild()->id,
            'annotation_id' => ImageAnnotationTest::create([
                'image_id' => ImageTest::create([
                    'volume_id' => $this->volume()->id,
                    'filename' => 'abc.jpg',
                ])->id,
            ])->id,
        ]);

        $params['oa_restrict_labels'] = [$this->labelRoot()->id];
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $params)
            // No annotations with the chosen label.
            ->assertStatus(422);

        $params['oa_restrict_labels'] = [$this->labelChild()->id];
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $params)
            ->assertSuccessful();
        $job = MaiaJob::first();
        $this->assertArrayHasKey('oa_restrict_labels', $job->params);
        $this->assertEquals([$this->labelChild()->id], $job->params['oa_restrict_labels']);
    }

    public function testStoreNdClustersTooFewImages()
    {
        $id = $this->volume()->id;
        $this->beEditor();
        $this->defaultParams['nd_clusters'] = 2;
        $this->postJson("/api/v1/volumes/{$id}/maia-jobs", $this->defaultParams)
            ->assertStatus(422);
    }

    public function testDestroy()
    {
        $job = MaiaJobTest::create(['volume_id' => $this->volume()->id]);
        $this->doTestApiRoute('DELETE', "/api/v1/maia-jobs/{$job->id}");

        $this->beGuest();
        $this->deleteJson("/api/v1/maia-jobs/{$job->id}")->assertStatus(403);

        $this->beEditor();
        // cannot be deleted during novelty detection
        $this->deleteJson("/api/v1/maia-jobs/{$job->id}")->assertStatus(422);

        $job->state_id = State::trainingProposalsId();
        $job->save();

        $this->deleteJson("/api/v1/maia-jobs/{$job->id}")->assertStatus(200);
        $this->assertNull($job->fresh());
    }
}
