import { Injectable } from '@angular/core';
import { HttpClient,HttpResponse} from '@angular/common/http';
import { range, from, Observable, throwError } from 'rxjs';
import { map, filter, scan,tap } from 'rxjs/operators';


import { HttpHeaders, HttpErrorResponse } from '@angular/common/http';



@Injectable()
export class LoginService {

    private loggedInStatus = false

    constructor(private http: HttpClient) { }

    setLoggedIn(value: boolean) {
        this.loggedInStatus = true;
        localStorage.setItem("loggedIn", "true")
    }

    isLoggedIn() {
        return JSON.parse(localStorage.getItem("currentUser")|| this.loggedInStatus.toString())    
    }


    logar(nome:string,senha:string){
        // let usuario = this.users.filter(x=>x.nome==nome && x.senha==senha)
        // console.log(usuario)
        // if(usuario){
        //     localStorage.setItem('currentUser', JSON.stringify(usuario));
        // }
        

    }
    // logarAPI(dn) {

        // const httpOptions = {
        //     headers: new HttpHeaders({
                // 'Authorization': 'Basic ' + btoa(JSON.parse(localStorage.getItem("currentUser"))['authToken'] + ':' + dn)                
        //     })
        // }
        

        // let bodyString = JSON.stringify({ "token": JSON.parse(localStorage.getItem("currentUser"))['authToken'], "dn": dn })
  
        // return this.http.post("http://localhost:8080/authenticate", bodyString,httpOptions).pipe(
        //     map((response: Headers) => {
        //         let body = response;
        //         if (body) {
        //             // localStorage.setItem('currentUser', JSON.stringify(body));
        //         }
        //         console.log("AQUI",body)
        //     }));
    // }



    // logar(passaporte: string, senha: string): Observable<any> {

    //     const httpOptions = {
    //         headers: new HttpHeaders({
    //             'Content-Type': 'application/json'
    //         })
    //     }
    //     let bodyString = JSON.stringify({ "passaporte": passaporte, "senha": senha })
        


        // return this.http.post('https://api.ufms.br/passaporte-ws/authentication', bodyString, httpOptions)
        //     .pipe(data);
      
    //     return this.http.post('https://api.ufms.br/passaporte-ws/authentication', bodyString, httpOptions)
    //         .pipe(
    //             map((response: any) => {
    //                 // console.log("ESSECARA", response);
    //                 let body = response;
    //                 if (body) {
    //                     localStorage.setItem('currentUser', JSON.stringify(body));
    //                 }
    //             }));
    // }

    // getPerfil(): Observable<any> {
    //     const httpOptions = {
    //         headers: new HttpHeaders({
    //             'Content-Type': 'application/json',
    //             'X-AUTH-TOKEN': JSON.parse(localStorage.getItem("currentUser"))['authToken']
    //         })
    //     }

    //     return this.http.get('https://api.ufms.br/passaporte-ws/ad/'+ JSON.parse(localStorage.getItem("currentUser"))['dn'] , httpOptions)
    //         .pipe(map((response: Response) => {
    //             localStorage.setItem('currentUserRole', JSON.stringify(response));
    //             return  JSON.parse(localStorage.getItem("currentUser"))['dn'];
    //         }));

    // }

    // updatePassword(senhaAtual, senhaNova): Observable<any> {
    //     var login = JSON.parse(localStorage.getItem("currentUser"))['usuario']['passaporte']
    //     const httpOptions = {
    //         headers: new HttpHeaders({
    //             'Content-Type': 'application/json',
    //             'X-AUTH-TOKEN': JSON.parse(localStorage.getItem("currentUser"))['authToken']
    //         })
    //     }

    //     let bodyString = JSON.stringify({ "login": login, "senhaAtual": senhaAtual, "senhaNova": senhaNova })
    //     return this.http.post('https://sistemas5.ufms.br/passaporte-ws/api/alterarSenha', bodyString, httpOptions)
    //         .pipe(map((response: Response) => {
    //             return response;
    //         }));
    // }


    logout() {
        localStorage.removeItem('currentUserRole')
        localStorage.removeItem('currentUser');
        this.loggedInStatus = false
    }
}