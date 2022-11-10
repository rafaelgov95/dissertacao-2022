import { Observable } from 'rxjs';
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { filter, map } from 'rxjs/operators';

import { Telefone, UsoTelefone } from './telefone-form/telefone';

@Component({
  selector: 'app-telefones',
  templateUrl: './telefones.component.html',
  styleUrls: ['./telefones.component.scss']
})
export class TelefonesComponent implements OnInit {


  telefones: Telefone[];

  constructor(private _http: HttpClient) { }

  ngOnInit() {
    // this.getTelefones()
    //   .subscribe(x => this.telefones = x);
  }

   adicionar() {
    var t: Telefone = new Telefone();
    t.id=4
    t.uso=1
    t.numero='67999507979'
    t.isWhatsApp=true
    this.telefones.push(t);
  }

  // private getTelefones(): Observable<Telefone[]> {
    // const url = 'assets/mock-telefones.json';
    // return this._http.get(url).pipe(map(x => x));
  // }

}
