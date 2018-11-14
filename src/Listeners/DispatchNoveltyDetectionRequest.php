<?php

namespace Biigle\Modules\Maia\Listeners;

use Queue;
use Exception;
use Illuminate\Contracts\Queue\ShouldQueue;
use Biigle\Modules\Maia\Events\MaiaJobCreated;
use Biigle\Modules\Maia\Jobs\NoveltyDetectionFailure;
use Biigle\Modules\Maia\Jobs\NoveltyDetectionRequest;

class DispatchNoveltyDetectionRequest implements ShouldQueue
{
   /**
     * Handle the event.
     *
     * @param  MaiaJobCreated  $event
     * @return void
     */
    public function handle(MaiaJobCreated $event)
    {
        $request = new NoveltyDetectionRequest($event->job);
        Queue::connection(config('maia.request_connection'))
            ->pushOn(config('maia.request_queue'), $request);
    }

    /**
     * Handle a job failure.
     *
     * @param  MaiaJobCreated  $event
     * @param  \Exception  $exception
     * @return void
     */
    public function failed(MaiaJobCreated $event, $exception)
    {
        $e = new Exception('The novelty detection request could not be submitted.');
        Queue::push(new NoveltyDetectionFailure($event->job->id, $e));
    }
}
