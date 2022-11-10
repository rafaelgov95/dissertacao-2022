import { DataCovid19 } from './../../modelos/DataCovid19';
import { DataECG } from './../../modelos/ECG';
import { HttpClient,HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';

import { from, Observable } from 'rxjs';
import { filter, map,flatMap } from 'rxjs/operators';
import { Data } from '@angular/router';
export interface PeriodicElement {
  name: string;
  api: string;
  data:number
}
@Injectable()
export class CNNWebApiService {
  base_url='http://127.0.0.1:8080'
  // base_url='http://cnnwebv2.ddns.net'
  constructor(private http: HttpClient) { }


    Diagnostico(arquivo:any,tipo:any,user:any): Observable<any> {
      console.log(user)
      const httpOptions = {
          headers: new HttpHeaders({
            'Content-Type':  'application/json'
          })
        };
      let bodyString = JSON.stringify({"name":user,"tipo":tipo,"conteudoArquivo":arquivo})
      return this.http.post(this.base_url+"/api", bodyString)
          .pipe(map((response: any) => {
              if (response) {
                  return response
              }
          }));
  }

  Get_Log(): Observable<PeriodicElement> {
    return this.http.get(this.base_url+"/logs")
        .pipe(map((response: any) => {
            if (response) {
                return response
            }
        }));
}

  downloadDataAsTxt(url: string): Observable<string> {

    return this.http.get(url,{ responseType: 'blob' }).pipe(
      flatMap(blob => {
        return this.DataAsTxt(blob);
      })
    );
  }

  downloadDataAsBase64(url: string): Observable<string> {
    return this.http.get(url, { responseType: 'blob' }).pipe(
      flatMap(blob => {
        return this.blobToBase64(blob);
      })
    );
  }

  private DataAsTxt(blob: any): Observable<any> {
    const fileReader = new FileReader();
    const observable = new Observable(observer => {
      fileReader.onloadend = () => {
        observer.next(fileReader.result);
        observer.complete();
      };
    });
    fileReader.readAsText(blob);
    return observable;
  }




  private blobToBase64(blob: Blob): Observable<any> {
    const fileReader = new FileReader();
    const observable = new Observable(observer => {
      fileReader.onloadend = () => {
        observer.next(fileReader.result);
        observer.complete();
      };
    });
    fileReader.readAsDataURL(blob);
    return observable;
  }



}
