import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import { MatInputModule } from '@angular/material/input';
import { CNNWebApiService } from '../../../shared/services/cnnapi/cnnapi';
import { PageHeaderModule } from '../../../shared/modules/page-header/page-header.module';
import { PipesModule } from '../../../shared/pipes/pipes-module';
import { ProcessamentoRoutingModule } from './processamento-routing.module';
import { ProcessamentoComponent } from './processamento.component';

import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';

import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatCardModule } from '@angular/material/card';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatSortModule } from '@angular/material/sort';
import { MatOptionModule } from '@angular/material/core';
import { MatSelectModule } from '@angular/material/select';
import * as PlotlyJS from 'plotly.js-dist-min';
import { PlotlyModule } from 'angular-plotly.js';

PlotlyModule.plotlyjs = PlotlyJS;
import { NgxLoadingModule, ngxLoadingAnimationTypes } from 'ngx-loading';


const MaterialAngularModule = [
  MatIconModule,
  MatButtonModule,
  MatButtonToggleModule,
  MatCardModule,
  MatSortModule,
  MatOptionModule,
  MatSelectModule,
  MatInputModule

]


@NgModule({
  imports: [
    CommonModule,
    HttpClientModule,
    ProcessamentoRoutingModule,
    FormsModule,
    ReactiveFormsModule,
    PipesModule,
    PageHeaderModule,
    MaterialAngularModule,
    PlotlyModule,
    NgbModule,
    NgxLoadingModule.forRoot({
      animationType: ngxLoadingAnimationTypes.wanderingCubes,
      backdropBackgroundColour: 'rgba(0,0,0,0.1)',
      backdropBorderRadius: '4px',
      primaryColour: '#ffffff',
      secondaryColour: '#ffffff',
      tertiaryColour: '#ffffff'
  })
  ],
  declarations: [ProcessamentoComponent],
  providers: [CNNWebApiService],
  exports: []
})
export class ProcessamentoModule { }
