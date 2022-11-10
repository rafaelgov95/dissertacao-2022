import { MatSortModule } from '@angular/material/sort';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { PageHeaderModule } from '../../../shared/modules/page-header/page-header.module';
import { TreinamentoRoutingModule } from './treinamento-routing.module';
import { TreinamentoComponent } from './treinamento.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { MatCardModule } from '@angular/material/card';
import { MatOptionModule } from '@angular/material/core';


const MaterialAngularModule = [  
  MatIconModule,
  MatButtonModule,
  MatButtonToggleModule,
  MatCardModule,
  MatSortModule,
  MatOptionModule,
]


@NgModule({
  imports: [
    CommonModule,
    TreinamentoRoutingModule,
     PageHeaderModule,
     MatCardModule,
     ReactiveFormsModule,
     FormsModule,
     HttpClientModule,
     MaterialAngularModule
  ],
  declarations: [TreinamentoComponent],
  providers: [],
  exports: [ ]
})
export class TreinamentoModule { }
