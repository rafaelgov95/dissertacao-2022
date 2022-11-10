import { HelpModule } from './help/help.module';
import { HomeComponent } from './home/home.component';
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { LayoutComponent } from './layout.component';

const routes: Routes = [
    {
        path: '', component: LayoutComponent,
        children: [
            { path: 'home', loadChildren: () => import('./home/home.module').then(m => m.HomeModule)}
            ,
            { path: 'processamento', loadChildren: () => import('./modulos/processamento/processamento.module').then(m => m.ProcessamentoModule)},
            // { path: 'treinamento', loadChildren: () => import('./modulos/treinamento/treinamento.module').then(m => m.TreinamentoModule)},
            // { path: 'repositorio', loadChildren: () => import('./modulos/repositorio/repositorio.module').then(m => m.RepositorioModule)},
            { path: 'gerenciador', loadChildren: () => import('./modulos/gerenciador/gerenciador.module').then(m => m.GerenciadorModule)},
            // { path: 'pseudo', loadChildren: () => import('./modulos/pseudo/pseudo.module').then(m => m.PseudoModule)},
            { path: 'help', loadChildren: () => import('./help/help.module').then(m => m.HelpModule)}

            ]
        }
]

@NgModule({
    imports: [RouterModule.forChild(routes)],
    exports: [RouterModule]
})
export class LayoutRoutingModule { }
