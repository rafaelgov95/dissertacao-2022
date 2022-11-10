import { from, Observable } from 'rxjs';
import { filter, map } from 'rxjs/operators';
import { Component, Input, Output, forwardRef, OnInit, EventEmitter, ViewContainerRef } from '@angular/core';
import { FormGroup, FormBuilder, Validators } from "@angular/forms";

@Component({
  selector: 'htmleditor-component',
  template: `
  
  `,

})
export class HtmleditorComponent implements OnInit {
  ckeditorContent
  ngOnInit() {
  }
}