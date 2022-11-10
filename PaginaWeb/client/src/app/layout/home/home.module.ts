import { PageHeaderModule } from '../../shared/modules/page-header/page-header.module';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';

import { HomeComponent } from './home.component';
import { HomeRoutingModule } from './home-routing.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';


@NgModule({
    imports: [
        CommonModule,
        HomeRoutingModule,
        PageHeaderModule,
         MatCardModule,
         MatInputModule
    ], exports: [
        PageHeaderModule
    ],
    declarations: [
        HomeComponent
    ],providers:[]
})
export class HomeModule { }
