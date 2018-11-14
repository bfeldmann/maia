<?php

namespace Biigle\Modules\Maia\Jobs;

use Queue;
use Exception;
use Biigle\Modules\Maia\MaiaJob;
use Biigle\Modules\Maia\MaiaJobState as State;
use Biigle\Modules\Maia\MaiaAnnotationType as Type;

class NoveltyDetectionResponse extends JobResponse
{
    /**
     * The job failed to process.
     *
     * @param  Exception  $exception
     * @return void
     */
    public function failed(Exception $exception)
    {
        Queue::push(new NoveltyDetectionFailure($this->jobId, $exception));
    }

    /**
     * {@inheritdoc}
     */
    protected function getNewAnnotationTypeId()
    {
        return Type::trainingProposalId();
    }

    /**
     * {@inheritdoc}
     */
    protected function getCreatedAnnotations(MaiaJob $job)
    {
        return $job->trainingProposals();
    }

    /**
     * {@inheritdoc}
     */
    protected function updateJobState(MaiaJob $job)
    {
        $job->state_id = State::trainingProposalsId();
        $job->save();
    }
}
