import { HelpComponent } from './help.component';
import { PageHeaderModule } from '../../shared/modules/page-header/page-header.module';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';

import { HelpRoutingModule } from './help-routing.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';


@NgModule({
    imports: [
        CommonModule,
        HelpRoutingModule,
        PageHeaderModule,
         MatCardModule,
         MatInputModule
    ], exports: [
        PageHeaderModule
    ],
    declarations: [
      HelpComponent
    ],providers:[]
})
export class HelpModule { }
