import { Rati } from './../../../modelos/rati-modelo';
import { HttpClient,HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { from, Observable } from 'rxjs';
import { filter, map } from 'rxjs/operators';

@Injectable()
export class RatiService {

    Rati: any
    constructor(private http: HttpClient) { }
   
    setRati(rati: any) {
        this.Rati = rati
    }
    getRati() {
        return this.Rati
    }

    getAllRati(): Observable<any> {
        const httpOptions = {
            headers: new HttpHeaders({
             
              'Authorization': 'Basic ' + btoa(JSON.parse(localStorage.getItem("currentUser"))['authToken'] + ':' + JSON.parse(localStorage.getItem("currentUser"))['dn'])            
            })
          };

        return this.http.get("http://localhost:8080/rati/",httpOptions)
            .pipe(map((response: Response) => {
              return response
            }));
    }


    AddRati(titulo, prioridade, mensagem,numero): Observable<any> {
     
        const httpOptions = {
            headers: new HttpHeaders({
              'Content-Type':  'application/json',
              'Authorization': 'Basic ' + btoa(JSON.parse(localStorage.getItem("currentUser"))['authToken'] + ':' + JSON.parse(localStorage.getItem("currentUserRole"))['dn'])

              // 'Authorization':  t.getItem("TokenAPI")
            })
          };

          let nome=JSON.parse(localStorage.getItem("currentUserRole"))['nomeCompleto']
          let cpf=JSON.parse(localStorage.getItem("currentUserRole"))['cpf']
          let bodyString = JSON.stringify({"nomeCompleto":nome,"cpf":cpf,"tipo":"fasdffasdaas","email":"rafaelgov95@gail.com", "titulo": titulo, "prioridade": prioridade, "mensagem": mensagem , "numero": numero})

        return this.http.post("http://localhost:8080/rati/", bodyString, httpOptions)
            .pipe(map((response: Response) => {
                if (response) {
                    return response
                }
            }));
    }


    AddResposta(body: Rati): Observable<any> {
       
      const httpOptions = {
        headers: new HttpHeaders({
          'Content-Type':  'application/json',
          'Authorization': 'Basic ' + btoa(JSON.parse(localStorage.getItem("currentUser"))['authToken'] + ':' + JSON.parse(localStorage.getItem("currentUser"))['dn'])                

          // 'Authorization':  sessionStorage.getItem("TokenAPI")
        })
      };


        let bodyString = JSON.stringify(body)

        return this.http.post("http://localhost:8080/rati/admin/resposta", bodyString, httpOptions)
            .pipe(map((response: Response) => {
                if (response) {
                    return response.json()
                }
            }));
    }
}