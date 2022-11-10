import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-treinamento',
  templateUrl: './repositorio.component.html',
  styleUrls: ['./repositorio.component.scss']
})
export class RepositorioComponent implements OnInit {

  imagens_covid_19 = [
    ['https://github.com/rafaelgov95/repotest/blob/master/COVID-19/covid-19-pneumonia-67.jpeg?raw=true','covid-19-pneumonia-67'],
    ['https://github.com/rafaelgov95/repotest/blob/master/COVID-19/covid-19-pneumonia-bilateral.jpg?raw=true','covid-19-pneumonia-bilateral'],
    ['https://github.com/rafaelgov95/repotest/blob/master/COVID-19/covid-19-pneumonia-mild.JPG?raw=true','covid-19-pneumonia-mild'],

    ['https://github.com/rafaelgov95/repotest/blob/master/COVID-19/d6494b39.jpg?raw=true','d6494b39'],
    ['https://github.com/rafaelgov95/repotest/blob/master/COVID-19/fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original.jpg?raw=true','fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original']


  ]
  constructor() { }

  ngOnInit(): void {
  }

}
