<?php

namespace Biigle\Modules\Maia\Http\Requests;

use Biigle\Modules\Maia\MaiaJob;
use Biigle\Modules\Maia\MaiaJobState as State;
use Biigle\Modules\Maia\Rules\OddNumber;
use Biigle\Volume;
use Illuminate\Foundation\Http\FormRequest;

class StoreMaiaJob extends FormRequest
{
    /**
     * The volume to create the MAIA job for.
     *
     * @var Volume
     */
    public $volume;

    /**
     * Determine if the user is authorized to make this request.
     *
     * @return bool
     */
    public function authorize()
    {
        $this->volume = Volume::findOrFail($this->route('id'));

        return $this->user()->can('edit-in', $this->volume);
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array
     */
    public function rules()
    {
        return [
            'training_data_method' => 'required|in:novelty_detection,own_annotations',
            'oa_restrict_labels' => 'array',
            'oa_restrict_labels.*' => 'integer|exists:labels,id',
            'nd_clusters' => 'required_if:training_data_method,novelty_detection|integer|min:1|max:100',
            'nd_patch_size' => ['required_if:training_data_method,novelty_detection', 'integer', 'min:3', 'max:99', new OddNumber],
            'nd_threshold' => 'required_if:training_data_method,novelty_detection|integer|min:0|max:99',
            'nd_latent_size' => 'required_if:training_data_method,novelty_detection|numeric|min:0.05|max:0.75',
            'nd_trainset_size' => 'required_if:training_data_method,novelty_detection|integer|min:1000|max:100000',
            'nd_epochs' => 'required_if:training_data_method,novelty_detection|integer|min:50|max:1000',
            'nd_stride' => 'required_if:training_data_method,novelty_detection|integer|min:1|max:10',
            'nd_ignore_radius' => 'required_if:training_data_method,novelty_detection|integer|min:0',
            'is_train_scheme' => 'required|array|min:1',
            'is_train_scheme.*' => 'array',
            'is_train_scheme.*.layers' => 'required|in:heads,all',
            'is_train_scheme.*.epochs' => 'required|integer|min:1',
            'is_train_scheme.*.learning_rate' => 'required|numeric|min:0|max:1',
        ];
    }

    /**
     * Configure the validator instance.
     *
     * @param  \Illuminate\Validation\Validator  $validator
     * @return void
     */
    public function withValidator($validator)
    {
        $validator->after(function ($validator) {
            if (!$this->volume->isImageVolume()) {
                $validator->errors()->add('volume', 'MAIA is only available for image volumes.');
            }

            $hasJobInProgress = MaiaJob::where('volume_id', $this->volume->id)
                ->whereIn('state_id', [
                    State::noveltyDetectionId(),
                    State::trainingProposalsId(),
                    State::instanceSegmentationId(),
                ])
                ->exists();

            if ($hasJobInProgress) {
                $validator->errors()->add('volume', 'A new MAIA job can only be sumbitted if there are no other jobs in progress for the same volume.');
            }

            if ($this->volume->hasTiledImages()) {
                $validator->errors()->add('volume', 'New MAIA jobs cannot be created for volumes with very large images.');
            }

            if ($this->input('training_data_method') === MaiaJob::TRAIN_NOVELTY_DETECTION && $this->volume->images()->count() < $this->input('nd_clusters')) {
                $validator->errors()->add('nd_clusters', 'The number of image clusters must not be greater than the number of images in the volume.');
            }
        });
    }
}
