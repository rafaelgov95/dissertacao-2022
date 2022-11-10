import { MatToolbarModule } from '@angular/material/toolbar';
import { MatPaginatorModule } from '@angular/material/paginator';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { CNNWebApiService } from '../../../shared/services/cnnapi/cnnapi';
import { GerenciadorComponent } from './gerenciador.component';
import { GerenciadorRoutingModule } from './gerenciador-routing.module';
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
import {MatTableModule} from '@angular/material/table';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import {MatTabsModule} from '@angular/material/tabs';


const MaterialAngularModule = [
  MatIconModule,
  MatButtonModule,
  MatButtonToggleModule,
  MatCardModule,
  MatSortModule,
  MatOptionModule,
  MatTableModule,
  MatFormFieldModule,
  MatInputModule,
  MatSortModule,
  MatPaginatorModule,
  MatProgressSpinnerModule,
  MatToolbarModule,
  MatTabsModule
]


@NgModule({
  imports: [
    CommonModule,
    GerenciadorRoutingModule,
     PageHeaderModule,
     MatCardModule,
     ReactiveFormsModule,
     FormsModule,
     HttpClientModule,
     MaterialAngularModule,
     NgbModule
  ],
  declarations: [GerenciadorComponent],
  providers: [CNNWebApiService],
  exports: [ ]
})
export class GerenciadorModule { }
