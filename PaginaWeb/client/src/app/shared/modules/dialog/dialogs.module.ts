import { DialogsService } from './/dialogs.service';
import { ConfirmDialog } from './confirm-dialog.component';
import { MatDialogModule, MatButtonModule  } from '@angular/material';
import { NgModule } from '@angular/core';



@NgModule({
    imports: [
        MatDialogModule,
        MatButtonModule,
    ],
    exports: [
        ConfirmDialog,
    ],
    declarations: [
        ConfirmDialog,
    ],
    providers: [
        DialogsService,
    ],
    entryComponents: [
        ConfirmDialog,
    ],
})
export class DialogsModule { }
