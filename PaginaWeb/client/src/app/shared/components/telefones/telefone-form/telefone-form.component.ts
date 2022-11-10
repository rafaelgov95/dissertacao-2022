import { Component, OnInit, OnChanges, Input, SimpleChanges } from '@angular/core';
import { FormControl, Validators } from '@angular/forms';
import { from, Observable,} from 'rxjs';

import { Telefone, UsoTelefone } from './telefone';

import { ViewEncapsulation } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';
import { MatIconRegistry } from '@angular/material';


@Component({
  selector: 'app-telefone-form',
  templateUrl: './telefone-form.component.html',
  styleUrls: ['./telefone-form.component.scss']
})

export class TelefoneFormComponent implements OnInit {

  @Input()
   telefone: Telefone;

   editando: boolean;

   usoControl = new FormControl('', [Validators.required]);
   usos = ['Pessoal', 'Trabalho', 'Ramal UFMS', 'Recado'];

   formControl: FormControl = new FormControl();

   operadoras = ['Algar Telecom', 'Claro', 'Nextel', 'Oi', 'Porto Seguro Conecta', 'TIM', 'Vivo'];
   operadorasFiltradas: Observable<string[]>;

  constructor(iconRegistry: MatIconRegistry, sanitizer: DomSanitizer) {
    // Registrando novos ícones
    iconRegistry.registerFontClassAlias('fontawesome', 'fa');
  }

  ngOnInit() {
    // this.operadorasFiltradas = this.formControl.valueChanges
    //   .startWith('')
    //   .map(val => this.filtrar(val));

    //   this.setEditar(false);
  }

  ngOnChanges(changes: SimpleChanges) {
  }

   filtrar(val: string): string[] {
    return this.operadoras.filter(option =>
      option.toLowerCase().indexOf(val.toLowerCase()) === 0);
  }

   isCelular(): boolean {
    var numeroStr: string = this.telefone.numero.replace(/[^0-9]/g, '');
    return numeroStr[2] == '9';
  }

   isRecado(): boolean {
    return this.telefone.uso.toString() === 'Recado';
  }

   setEditar(b: boolean) {
    this.editando = b;
    if (this.editando) {
      this.usoControl.enable();
      this.formControl.enable();
    } else {
      this.usoControl.disable();
      this.formControl.disable();
    }
  }

   editar() {
    this.setEditar(!this.editando);
  }

   salvar() {
    this.editar();
  }

   floatPlaceholderValue(): string {
    if (!this.editando) {
      return 'always';
    }
    return 'auto';
  }

   mask(userInput) {
    let numbers = userInput.match(/\d/g);
    let numberLength = 0;
    if (numbers) {
      numberLength = numbers.join("").length;
    }

    if (numberLength > 10) {
      return ['(', /[1-9]/, /[1-9]/, ')', ' ', /\d/, /\d/, /\d/, /\d/, /\d/, '-', /\d/, /\d/, /\d/, /\d/];
    } else {
      return ['(', /[1-9]/, /[1-9]/, ')', ' ', /\d/, /\d/, /\d/, /\d/, '-', /\d/, /\d/, /\d/, /\d/];
    }
  }

}
