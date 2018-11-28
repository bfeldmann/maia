biigle.$viewModel("maia-job-form",function(e){new Vue({el:e,data:{showAdvanced:!1},methods:{toggle:function(){this.showAdvanced=!this.showAdvanced}}})}),biigle.$viewModel("maia-show-container",function(e){var t=biigle.$require("maia.job"),i=biigle.$require("maia.states"),n=biigle.$require("maia.api.maiaJob"),a=biigle.$require("maia.api.maiaAnnotation"),s=biigle.$require("messages.store"),o=biigle.$require("annotations.stores.images"),r=[],c={};new Vue({el:e,mixins:[biigle.$require("core.mixins.loader")],components:{sidebar:biigle.$require("annotations.components.sidebar"),sidebarTab:biigle.$require("core.components.sidebarTab"),selectTpTab:biigle.$require("maia.components.selectTpTab"),tpImageGrid:biigle.$require("maia.components.tpImageGrid"),refineTpTab:biigle.$require("maia.components.refineTpTab"),refineTpCanvas:biigle.$require("maia.components.refineTpCanvas"),reviewAcTab:biigle.$require("maia.components.reviewAcTab"),acImageGrid:biigle.$require("maia.components.acImageGrid")},data:{visitedSelectTpTab:!1,visitedRefineTpTab:!1,visitedReviewAcTab:!1,openTab:"info",hasTrainingProposals:!1,selectedTrainingProposalIds:{},lastSelectedTp:null,currentImage:null,currentTpIndex:0,annotationCandidates:[]},computed:{infoTabOpen:function(){return"info"===this.openTab},selectTpTabOpen:function(){return"select-training-proposals"===this.openTab},refineTpTabOpen:function(){return"refine-training-proposals"===this.openTab},reviewAcTabOpen:function(){return"review-annotation-candidates"===this.openTab},isInTrainingProposalState:function(){return t.state_id===i["training-proposals"]},isInAnnotationCandidateState:function(){return t.state_id===i["annotation-candidates"]},trainingProposals:function(){return this.hasTrainingProposals?r:[]},selectedTrainingProposals:function(){return Object.keys(this.selectedTrainingProposalIds).map(function(e){return c[e]})},hasNoSelectedTp:function(){return 0===this.selectedTrainingProposals.length},visitedSelectOrRefineTpTab:function(){return this.visitedSelectTpTab||this.visitedRefineTpTab}},methods:{handleSidebarToggle:function(){this.$nextTick(function(){this.$refs.imageGrid.$emit("resize")})},handleTabOpened:function(e){this.openTab=e},setTrainingProposals:function(e){var t=this;r=e.body,r.forEach(function(e){c[e.id]=e,t.setSelectedTrainingProposalId(e)}),this.hasTrainingProposals=r.length>0},fetchTrainingProposals:function(){return this.startLoading(),n.getTrainingProposals({id:t.id}).then(this.setTrainingProposals).catch(s.handleErrorResponse).finally(this.finishLoading)},openRefineTpTab:function(){this.openTab="refine-training-proposals"},handleSelectedTrainingProposal:function(e,t){e.selected?this.updateSelectTrainingProposal(e,!1):t.shiftKey&&this.lastSelectedTp?this.selectAllTpBetween(e,this.lastSelectedTp):(this.lastSelectedTp=e,this.updateSelectTrainingProposal(e,!0))},selectAllTpBetween:function(e,t){var i=this.trainingProposals.indexOf(e),n=this.trainingProposals.indexOf(t);if(n<i){var a=n;n=i,i=a}for(var s=i+1;s<=n;s++)this.updateSelectTrainingProposal(this.trainingProposals[s],!0)},updateSelectTrainingProposal:function(e,t){e.selected=t,this.setSelectedTrainingProposalId(e),a.update({id:e.id},{selected:t}).catch(function(i){s.handleErrorResponse(i),e.selected=!t,this.setSelectedTrainingProposalId(e)})},setSelectedTrainingProposalId:function(e){e.selected?Vue.set(this.selectedTrainingProposalIds,e.id,!0):Vue.delete(this.selectedTrainingProposalIds,e.id)},fetchCurrentImage:function(){return this.startLoading(),o.fetchAndDrawImage(this.currentImageId).catch(function(e){s.danger(e)}).then(this.setCurrentImage).then(this.cacheNextImage).finally(this.finishLoading)},setCurrentImage:function(e){this.currentImage=e},cacheNextImage:function(){this.currentImageId!==this.nextImageId&&o.fetchImage(this.nextImageId).catch(function(){})},handleNext:function(){this.currentTp&&(this.currentTp.seen=!0),this.currentTpIndex=this.nextTpIndex},handlePrevious:function(){this.currentTp&&(this.currentTp.seen=!0),this.currentTpIndex=this.previousTpIndex},handleRefineTp:function(e){Vue.Promise.all(e.map(this.updateTpPoints)).catch(s.handleErrorResponse)},updateTpPoints:function(e){var t=this;return a.update({id:e.id},{points:e.points}).then(function(){t.tpById[e.id].points=e.points})},focusCurrentTp:function(){this.$refs.refineCanvas.focusAnnotation(this.currentTp,!0,!1)},handleSelectTp:function(e){this.updateSelectTrainingProposal(e,!0)},handleUnselectTp:function(e){this.updateSelectTrainingProposal(e,!1)}},watch:{selectTpTabOpen:function(e){this.visitedSelectTpTab=!0,e&&biigle.$require("keyboard").setActiveSet("select-tp")},refineTpTabOpen:function(e){this.visitedRefineTpTab=!0,e&&biigle.$require("keyboard").setActiveSet("refine-tp")},reviewAcTabOpen:function(e){this.visitedReviewAcTab=!0,open&&biigle.$require("keyboard").setActiveSet("review-ac")},visitedSelectOrRefineTpTab:function(){this.fetchTrainingProposals()},visitedReviewAcTab:function(){this.fetchAnnotationCandidates()},currentImageId:function(e){e?this.fetchCurrentImage().then(this.focusCurrentTp):this.setCurrentImage(null)},currentTp:function(e){e&&this.focusCurrentTp()}}})}),biigle.$declare("maia.api.maiaAnnotation",Vue.resource("api/v1/maia-annotations{/id}",{},{getFile:{method:"GET",url:"api/v1/maia-annotations{/id}/file"}})),biigle.$declare("maia.api.maiaJob",Vue.resource("api/v1/maia-jobs{/id}",{},{save:{method:"POST",url:"api/v1/volumes{/id}/maia-jobs"},getTrainingProposals:{method:"GET",url:"api/v1/maia-jobs{/id}/training-proposals"},getAnnotationCandidates:{method:"GET",url:"api/v1/maia-jobs{/id}/annotation-candidates"}})),biigle.$component("maia.components.acImageGrid",{mixins:[biigle.$require("volumes.components.imageGrid")],components:{imageGridImage:biigle.$require("maia.components.acImageGridImage")}}),biigle.$component("maia.components.acImageGridImage",{mixins:[biigle.$require("volumes.components.imageGridImage")],template:'<figure class="image-grid__image" :class="classObject" :title="title"><div v-if="showIcon" class="image-icon"><i class="fas fa-3x" :class="iconClass"></i></div><img @click="toggleSelect" :src="url || emptyUrl"><div v-if="showAnnotationLink" class="image-buttons"><a :href="showAnnotationLink" target="_blank" class="image-button" title="Show the annotation in the annotation tool"><span class="fa fa-external-link-square-alt" aria-hidden="true"></span></a></div></figure>',computed:{showAnnotationLink:function(){return!1},selected:function(){return this.image.selected},title:function(){return this.selected?"Remove selected label":"Assign selected label"}},methods:{getBlob:function(){return biigle.$require("maia.api.maiaAnnotation").getFile({id:this.image.id})}}}),biigle.$component("maia.components.refineTpCanvas",{mixins:[biigle.$require("annotations.components.annotationCanvas")],props:{unselectedAnnotations:{type:Array,default:function(){return[]}}},data:function(){return{selectingTp:!1}},computed:{},methods:{toggleMarkAsInteresting:function(){this.selectingTp=!this.selectingTp},createUnselectedAnnotationsLayer:function(){this.unselectedAnnotationFeatures=new ol.Collection,this.unselectedAnnotationSource=new ol.source.Vector({features:this.unselectedAnnotationFeatures}),this.unselectedAnnotationLayer=new ol.layer.Vector({source:this.unselectedAnnotationSource,zIndex:50,updateWhileAnimating:!0,updateWhileInteracting:!0,style:biigle.$require("annotations.stores.styles").editing,opacity:.5})},createSelectTpInteraction:function(e){var t=biigle.$require("annotations.ol.AttachLabelInteraction");this.selectTpInteraction=new t({map:this.map,features:e}),this.selectTpInteraction.setActive(!1),this.selectTpInteraction.on("attach",this.handleSelectTp)},handleSelectTp:function(e){this.$emit("select-tp",e.feature.get("annotation"))},handleUnselectTp:function(){this.selectedAnnotations.length>0&&this.$emit("unselect-tp",this.selectedAnnotations[0])}},watch:{unselectedAnnotations:function(e){this.refreshAnnotationSource(e,this.unselectedAnnotationSource)},selectingTp:function(e){this.selectTpInteraction.setActive(e)}},created:function(){this.createUnselectedAnnotationsLayer(),this.map.addLayer(this.unselectedAnnotationLayer),this.selectInteraction.setActive(!1),this.canModify&&(this.createSelectTpInteraction(this.unselectedAnnotationFeatures),this.map.addInteraction(this.selectTpInteraction),biigle.$require("keyboard").on("Delete",this.handleUnselectTp,0,this.listenerSet))}}),biigle.$component("maia.components.refineTpTab",{props:{selectedTrainingProposals:{type:Array,required:!0},seenTrainingProposals:{type:Array,required:!0}},data:function(){return{}},computed:{numberSelectedTps:function(){return this.selectedTrainingProposals.length},numberSeenTps:function(){return this.seenTrainingProposals.length},hasNoSelectedTp:function(){return 0===this.numberSelectedTps},hasSeenAllSelectedTps:function(){return this.numberSelectedTps>0&&this.numberSelectedTps===this.numberSeenTps},textClass:function(){return this.hasSeenAllSelectedTps?"text-success":""},buttonClass:function(){return this.hasSeenAllSelectedTps?"btn-success":"btn-default"}},methods:{},created:function(){}}),biigle.$component("maia.components.reviewAcTab",{props:{},data:function(){return{}},computed:{},methods:{},created:function(){}}),biigle.$component("maia.components.selectTpTab",{props:{trainingProposals:{type:Array,required:!0},selectedTrainingProposals:{type:Array,required:!0}},data:function(){return{}},computed:{selectedTpCount:function(){return this.selectedTrainingProposals.length},tpCount:function(){return this.trainingProposals.length}},methods:{proceed:function(){this.$emit("proceed")}},created:function(){}}),biigle.$component("maia.components.tpImageGrid",{mixins:[biigle.$require("volumes.components.imageGrid")],components:{imageGridImage:biigle.$require("maia.components.tpImageGridImage")},props:{selectedTpIds:{type:Object,required:!0}}}),biigle.$component("maia.components.tpImageGridImage",{mixins:[biigle.$require("volumes.components.imageGridImage")],template:'<figure class="image-grid__image" :class="classObject" :title="title"><div v-if="showIcon" class="image-icon"><i class="fas fa-3x" :class="iconClass"></i></div><img @click="toggleSelect" :src="url || emptyUrl"><div v-if="showAnnotationLink" class="image-buttons"><a :href="showAnnotationLink" target="_blank" class="image-button" title="Show the annotation in the annotation tool"><span class="fa fa-external-link-square-alt" aria-hidden="true"></span></a></div></figure>',computed:{showAnnotationLink:function(){return!1},selected:function(){return this.$parent.selectedTpIds.hasOwnProperty(this.image.id)},title:function(){if(this.selectable)return this.selected?"Undo marking as interesting":"Mark as interesting"}},methods:{getBlob:function(){return biigle.$require("maia.api.maiaAnnotation").getFile({id:this.image.id})}}});