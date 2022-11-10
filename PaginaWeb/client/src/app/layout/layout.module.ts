
// import { NgxLoadingModule } from 'ngx-loading';
import { MatIconModule } from '@angular/material/icon';
import { PageHeaderModule } from '../shared/modules/page-header/page-header.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgbDropdownModule } from '@ng-bootstrap/ng-bootstrap';
import { LayoutRoutingModule } from './layout-routing.module';
import { LayoutComponent } from './layout.component';
import { HeaderComponent, SidebarComponent } from '../shared';

import {MatToolbarModule} from '@angular/material/toolbar';


@NgModule({
    imports: [
        CommonModule,
        NgbDropdownModule,
        LayoutRoutingModule,
        PageHeaderModule,
        MatToolbarModule,
        MatIconModule,
        // NgxLoadingModule.forRoot({})
        ],
    declarations: [
        LayoutComponent,
        HeaderComponent,
        SidebarComponent
    ],
    exports: [ ]
})
export class LayoutModule { }
