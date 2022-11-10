import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { LoginService } from './../shared/services/login/login-service';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { LoginRoutingModule } from './login-routing.module';
import { LoginComponent } from './login.component';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
// import { ToastrModule } from 'ngx-toastr';
import { BrowserModule } from '@angular/platform-browser';


@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    LoginRoutingModule,
    ReactiveFormsModule,
    MatProgressSpinnerModule

  ],
  providers: [],
  declarations: [LoginComponent,
  ],
  exports: [LoginComponent]
})
export class LoginModule { }
