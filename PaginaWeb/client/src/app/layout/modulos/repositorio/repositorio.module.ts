import { RepositorioComponent } from './repositorio.component';
import { RepositorioRoutingModule } from './repositorio-routing.module';
import { MatSortModule } from '@angular/material/sort';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { PageHeaderModule } from '../../../shared/modules/page-header/page-header.module';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClientModule } from '@angular/common/http';
import { MatCardModule } from '@angular/material/card';
import { MatOptionModule } from '@angular/material/core';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';


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
    RepositorioRoutingModule,
     PageHeaderModule,
     MatCardModule,
     ReactiveFormsModule,
     FormsModule,
     HttpClientModule,
     MaterialAngularModule,
     NgbModule
  ],
  declarations: [RepositorioComponent],
  providers: [],
  exports: [ ]
})
export class RepositorioModule { }
