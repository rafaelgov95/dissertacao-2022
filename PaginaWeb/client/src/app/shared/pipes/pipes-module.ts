import { SafeHtmlPipe } from './htmlview';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

@NgModule({
    imports: [
        CommonModule
    ],
    declarations: [SafeHtmlPipe],
    exports:[SafeHtmlPipe]
})
export class PipesModule { }