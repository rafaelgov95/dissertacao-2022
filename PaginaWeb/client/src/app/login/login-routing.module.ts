import { LayoutModule } from '../layout/layout.module';
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { LoginComponent } from './login.component';

const routes: Routes = [
    { path: '', component: LoginComponent }

    // { path: '', loadChildren: () => import('./LayoutModule').then(m => m.LayoutModule)}

];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class LoginRoutingModule { }
