import { HttpClient } from '@angular/common/http';
import { MatTableDataSource } from '@angular/material/table';
import { CNNWebApiService } from '../../../shared/services/cnnapi/cnnapi';
import { Component, OnInit, ViewChild, AfterViewInit } from '@angular/core';
import { MatPaginator } from '@angular/material/paginator';
import { MatSort } from '@angular/material/sort';
import { Data } from '@angular/router';
import { merge, Observable, of as observableOf } from 'rxjs';
import { catchError, map, startWith, switchMap } from 'rxjs/operators';

export interface PeriodicElement {
  name: string;
  Processamento: string;
  data: Data
}
var ELEMENT_DATA: PeriodicElement[]



@Component({
  selector: 'app-gerenciador',
  templateUrl: './gerenciador.component.html',
  styleUrls: ['./gerenciador.component.scss']
})
export class GerenciadorComponent implements OnInit {
  dataSource:any;
  resultsLength = 0;
  isLoadingResults = true;
  isRateLimitReached = false;
  displayedColumns: string[] = ['posicao', 'name', 'api', 'data'];
  displayedColumns_count: string[] = ['posicao', 'name', 'number'];

  @ViewChild(MatPaginator) paginator!: MatPaginator;
  @ViewChild(MatSort, {static: true}) sort!: MatSort;
  exampleDatabase!: ExampleHttpDatabase | null;
  data: GithubIssue[] = [];
  constructor(private cnnwebapiservice: CNNWebApiService, private _httpClient: HttpClient) { }


  // ngAfterViewInit() {
  //   this.exampleDatabase = new ExampleHttpDatabase(this._httpClient);

  //   // If the user changes the sort order, reset back to the first page.
  //   this.sort.sortChange.subscribe(() => this.paginator.pageIndex = 0);

  //   merge(this.sort.sortChange, this.paginator.page)
  //     .pipe(
  //       startWith({}),
  //       switchMap(() => {
  //         this.isLoadingResults = true;
  //         return this.exampleDatabase!.getRepoIssues(
  //           this.sort.active, this.sort.direction, this.paginator.pageIndex);
  //       }),
  //       map(data => {
  //         // Flip flag to show that loading has finished.
  //         this.isLoadingResults = false;
  //         this.isRateLimitReached = false;
  //         this.resultsLength = data.total_count;

  //         return data.items;
  //       }),
  //       catchError(() => {
  //         this.isLoadingResults = false;
  //         // Catch if the GitHub API has reached its rate limit. Return empty data.
  //         this.isRateLimitReached = true;
  //         return observableOf([]);
  //       })
  //     ).subscribe(data => this.data = data);
  // }



  dataSource_cont!:any
  ngOnInit(): void {
    this.cnnwebapiservice.Get_Log().subscribe(data => {
      // console.log(data['logs'])
      this.dataSource = data
      this.dataSource.reverse()
      var array_temp=[];

      const set_contador = new Set();
      var occurrences:any =[];
      var array_new =[];
      for (var i = 0, j = this.dataSource.length; i < j; i++) {
         occurrences[this.dataSource[i]['name']] = (occurrences[this.dataSource[i]['name']] || 0) + 1;
      }
      // console.log(occurrences["rafael"])

    //   for (var k = 0, b = occurrences.length; k < b; k++) {
    //     // array_new[]
    //     console.log(occurrences[occurrences[k]])


    //  }
      // console.log(occurrences)
      for (let person in occurrences) {
        const v = {"name":person,"value":occurrences[person]}
        array_new.push(v)
      }
      array_new.sort(function(a, b) {
        return b.value - a.value;
      });


      this.dataSource_cont=array_new
      // this.dataSource_cont = new MatTableDataSource(this.dataSource_cont)
      this.dataSource = new MatTableDataSource(this.dataSource)
      this.dataSource.sort = this.sort;

    }, error => {

    })

  }

  applyFilter(event: Event) {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();
  }



}
export interface Person {
  firstName: string;
  value: number;
}
export interface GithubApi {
  items: GithubIssue[];
  total_count: number;
}

export interface GithubIssue {
  created_at: string;
  number: string;
  state: string;
  title: string;
}

export class ExampleHttpDatabase {
  constructor(private _httpClient: HttpClient) { }

  getRepoIssues(sort: string, order: string, page: number): Observable<GithubApi> {
    const href = 'https://api.github.com/search/issues';
    const requestUrl =
      `${href}?q=repo:angular/components&sort=${sort}&order=${order}&page=${page + 1}`;

    return this._httpClient.get<GithubApi>(requestUrl);
  }
}
