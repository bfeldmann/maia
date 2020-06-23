/**
 * A letiant of the image grid image used for the selection of MAIA annotation candidates.
 *
 * @type {Object}
 */
biigle.$component('maia.components.candidatesImageGridImage', {
    mixins: [
        biigle.$require('volumes.components.imageGridImage'),
        biigle.$require('largo.mixins.annotationPatch'),
    ],
    template: '<figure class="image-grid__image image-grid__image--annotation-candidate" :class="classObject" :title="title">' +
        '<div v-if="showIcon" class="image-icon">' +
            '<i class="fas" :class="iconClass"></i>' +
        '</div>' +
        '<img @click="toggleSelect" :src="url || emptyUrl" @error="showEmptyImage">' +
        '<div v-if="showAnnotationLink" class="image-buttons">' +
            '<a :href="showAnnotationLink" target="_blank" class="image-button" title="Show the annotation in the annotation tool">' +
                '<span class="fa fa-external-link-square-alt" aria-hidden="true"></span>' +
            '</a>' +
        '</div>' +
        '<div v-if="selected" class="attached-label">' +
            '<span class="attached-label__color" :style="labelStyle"></span> ' +
            '<span class="attached-label__name" v-text="label.name"></span>' +
        '</div>' +
    '</figure>',
    computed: {
        showAnnotationLink() {
            return false;
        },
        label() {
            if (this.selected) {
                return this.$parent.selectedCandidateIds[this.image.id];
            }

            return null;
        },
        selected() {
            return this.$parent.selectedCandidateIds.hasOwnProperty(this.image.id);
        },
        converted() {
            return this.$parent.convertedCandidateIds.hasOwnProperty(this.image.id);
        },
        selectable() {
            return !this.converted;
        },
        classObject() {
            return {
                'image-grid__image--selected': this.selected || this.converted,
                'image-grid__image--selectable': this.selectable,
                'image-grid__image--fade': this.selectedFade,
                'image-grid__image--small-icon': this.smallIcon,
            };
        },
        iconClass() {
            if (this.converted) {
                return 'fa-lock';
            }

            return 'fa-' + this.selectedIcon;
        },
        showIcon() {
            return this.selectable || this.selected || this.converted;
        },
        title() {
            if (this.converted) {
                return 'This annotation candidate has been converted';
            }

            return this.selected ? 'Detach label' : 'Attach selected label';
        },
        labelStyle() {
            return {
                'background-color': '#' + this.label.color,
            };
        },
        id() {
            return this.image.id;
        },
        uuid() {
            return this.image.uuid;
        },
        urlTemplate() {
            return biigle.$require('maia.acUrlTemplate');
        },
    },
});
