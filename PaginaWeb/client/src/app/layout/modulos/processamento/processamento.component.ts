import { DomSanitizer } from '@angular/platform-browser';
import { ngxLoadingAnimationTypes,NgxLoadingComponent } from 'ngx-loading';
import { HttpClient } from '@angular/common/http';
import { Observable, Observer } from 'rxjs';
import { ToastrService } from 'ngx-toastr';
import { DataECG } from '../../../shared/modelos/ECG';
import { CNNWebApiService } from '../../../shared/services/cnnapi/cnnapi';
import { Component, OnInit,ViewChild, TemplateRef } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from "@angular/forms";
import { DataCovid19 } from 'src/app/shared/modelos/DataCovid19';
import { map, flatMap } from 'rxjs/operators';
import { NgbCarousel } from '@ng-bootstrap/ng-bootstrap';

interface Diagnostico {
  tipo: string;
  shape: String;
}
const PrimaryWhite = '#ffffff';
const SecondaryGrey = '#ccc';
const PrimaryRed = '#dd0031';
const SecondaryBlue = '#006ddd';
@Component({
  selector: 'app-processamento',
  templateUrl: './processamento.component.html',
  styleUrls: ['./processamento.component.scss']
})
export class ProcessamentoComponent implements OnInit {
  @ViewChild('ngxLoading', { static: false })
  ngxLoadingComponent!: NgxLoadingComponent;

  @ViewChild('customLoadingTemplate', { static: false }) customLoadingTemplate!: TemplateRef<any>;
  public quokkaAsyncData!: Observable<string>;
  public quokkaData!: string;
  public loadingTemplate!: TemplateRef<any>;
  public primaryColour = PrimaryWhite;
  public secondaryColour = SecondaryGrey;
  public ngxLoadingAnimationTypes = ngxLoadingAnimationTypes;
  public config = { animationType: ngxLoadingAnimationTypes.none, primaryColour: this.primaryColour, secondaryColour: this.secondaryColour, tertiaryColour: this.primaryColour, backdropBorderRadius: '3px' };
  public graph1 = {
    data: [
      { x: [], y: [], type: 'scatter', mode: 'lines+points', marker: { color: 'red' }, name:"Sem filtro" },
      { x: [], y: [], type: 'scatter', mode: 'lines+points', marker: { color: 'blue' }, name:"Com filtro" }
    ],
    layout: { width: 640, height: 550, title: 'Original - RR intervals (ms)' }
  };
  public graph2 = {
    data: [
      { x: [], y: [], type: 'scatter', mode: 'lines+points', marker: { color: 'red' }, }
    ],
    layout: { width: 640, height: 550, title: 'Filtrado - RR intervals (ms)' }
  };
  public graph3 = {
    data: [
      { z: [], type: 'surface', showscale: false }
    ],
    layout: { width: 640, height: 550, title: 'Wavelet Surface RR' }
  };

  public graph_histograma_covid_antes = {

    data: [
      {
         x:[],
         y:[],
        type: 'bar',
      }
    ],
    layout: { width: 640, height: 550, title: 'Histograma Original' }
  };


  diagnosticos: Diagnostico[] = [
    { tipo: 'VFC', shape: '.txt' }

  ];
  demonstracao=false;
  file!: File
  show_resultdo = false;
  loading_value = false;
  show_vfc = false
  dataECG!: DataECG;

  paused = false;
  mostrar_vfc=false;

  uploadForm: FormGroup;
  dados_vfc!: String;

  @ViewChild('carousel', {static : true}) carousel!: NgbCarousel;

  constructor(private sanitizer: DomSanitizer,
    // private toastr: ToastrService,
     public fb: FormBuilder, private cnnwebapiservice: CNNWebApiService) {
    this.uploadForm = this.fb.group({
      'diagnostico': ['',
        Validators.required
      ],
      'dados_input': ['', [
        Validators.required
      ]],
      'valido': [false, [
        Validators.requiredTrue
      ]],
    });

  }
  arquivo_vfc_atual=''
  arquivo_vfc_atual_index=0
  arquivos_vfc_original=[
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/VFC/0-1.txt','0-1.txt'],
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/VFC/1-1.txt','1-1.txt'],
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/VFC/2-1.txt','2-1.txt']
  ]
  arquivos_vfc:any= []
  ngOnInit() {
    // this.loadingTemplate = this.customLoadingTemplate;

   }

  ngOnDestroy() {
    this
  }


  checkFile() {
    console.log("Check Filse")
    this.mostrar_vfc=false
    if (this.uploadForm.value.diagnostico.tipo == "VFC"){
    for (let index = 0; index < this.arquivos_vfc_original.length; index++) {
      const element = this.arquivos_vfc_original[index][0];
          this.downloadDataAsTxt(element,index)
          console.log(index)
      }
    }
    if (this.file) {
     if ((this.uploadForm.value.diagnostico.tipo == "VFC") && (this.file.type == "text/plain")) {
        // this.FormshowSuccess()
        this.uploadForm.controls['valido'].setValue(true)
      } else if (this.uploadForm.value.diagnostico.tipo == "VFC") {
        // this.FormatError("VFC")
        this.uploadForm.controls['valido'].setValue(false)
      }
    } else {
      this.uploadForm.controls['valido'].setValue(false)
      this.uploadForm.controls['dados_input'].reset()
      this.show_resultdo = false
      this.show_vfc = false
     }
  }

  showPreview(event:any) {
    const file_local = (event.target as HTMLInputElement).files![0]
    console.log(file_local)
    if (file_local) {
      this.file = file_local
      this.uploadForm.controls['dados_input'].reset()
      console.log("Change File")
      this.show_resultdo = false
      this.show_vfc = false

      const reader = new FileReader();

      if (file_local.type == "text/plain") {
        reader.readAsText(file_local)
        reader.onload = () => {
          const valor: string = reader.result as string
          this.uploadForm.controls['dados_input'].setValue(valor)
          // this.uploadForm.controls['dados_input'].setValue(JSON.stringify(valor))
        }
      } else {
        // this.FormatFileInvalid(this.file.type)
      }

    }
    console.log(this.uploadForm.controls['dados_input'])

  }
  uploadFormPreview=true
  submit() {

    if (this.uploadForm.valid) {
      this.loading_value = true;
      let user = JSON.parse(localStorage.getItem("currentUser")|| '{}')[0];
     if (this.uploadForm.value.diagnostico.tipo == "VFC") {
        this.cnnwebapiservice.Diagnostico(this.uploadForm.controls['dados_input'].value, "VFC",user.nome).subscribe(
          data => {
            this.loading_value = false;
            this.dataECG = data
            this.graph1.data[0].x = this.dataECG.Plot_Original_x_axis
            this.graph1.data[0].y = this.dataECG.Plot_Original_y_axis

            this.graph1.data[1].x = this.dataECG.Plot_Filtrada_x_axis
            this.graph1.data[1].y = this.dataECG.Plot_Filtrada_y_axis

            this.graph2.data[0].x = this.dataECG.Plot_Filtrada_x_axis
            this.graph2.data[0].y = this.dataECG.Plot_Filtrada_y_axis

            this.graph3.data[0].z = this.dataECG.Plot_Wavelet

//            this.showSuccess()
            this.show_resultdo = true
            this.show_vfc = true;
          },
          error => {
//            this.FormatAPI();
            this.loading_value = false;
            this.show_vfc = false;
          });
          this.mostrar_vfc=!this.mostrar_vfc
      }


    }

  }
  // FormshowSuccess() {
  //   this.toastr.success('Carregados com Sucesso!', 'Dados de Entrada');
  // }
  // showSuccess() {
  //   this.toastr.success('Sucesso!', 'Request');
  // }
  // FormatError(tipo) {
  //   this.toastr.error('Formato inválido para o tipo ' + tipo, 'Dado de Entrada');
  // }

  // FormatAPI() {
  //   this.toastr.error('Erro na requisição ', 'API CNNWEB');
  // }

  // FormatFileInvalid(tipo) {
  //   this.toastr.error('Tipo ' + tipo + ' file inválido ', 'Dado de Entrada');
  // }

  imagens_covid_19 = [
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/COVID-19/covid-19-pneumonia-67.jpeg','covid-19-pneumonia-67.jpeg'],
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/COVID-19/covid-19-pneumonia-bilateral.jpg','covid-19-pneumonia-bilateral.jpeg'],
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/COVID-19/covid-19-pneumonia-mild.JPG','covid-19-pneumonia-mild.jpeg'],

    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/COVID-19/d6494b39.jpg','d6494b39.jpeg'],
    ['https://raw.githubusercontent.com/rafaelgov95/repotest/master/COVID-19/fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original.jpg','fff49165-b22d-4bb4-b9d1-d5d62c52436c.annot.original.jpeg']

  ]



  mostrar_teste(){
  if(this.uploadForm.controls['diagnostico'].value.tipo=='VFC'){

      this.arquivo_vfc_atual = this.arquivos_vfc[0]
      this.mostrar_vfc =!this.mostrar_vfc

    }

  }



  proximo_vfc(){
    console.log((this.arquivo_vfc_atual_index) )

    if((this.arquivo_vfc_atual_index+1) >= this.arquivos_vfc.length){
      this.arquivo_vfc_atual = this.arquivos_vfc[0]
      this.arquivo_vfc_atual_index = 0

    }else{
      this.arquivo_vfc_atual_index = this.arquivo_vfc_atual_index+1

      this.arquivo_vfc_atual = this.arquivos_vfc[this.arquivo_vfc_atual_index]
    }

  }
  anterior_vfc(){
    console.log(this.arquivos_vfc)
    console.log((this.arquivo_vfc_atual_index) )
    if((this.arquivo_vfc_atual_index-1) >= 0){
      this.arquivo_vfc_atual_index = this.arquivo_vfc_atual_index-1

      this.arquivo_vfc_atual = this.arquivos_vfc[(this.arquivo_vfc_atual_index)]

    }else{
      this.arquivo_vfc_atual_index = 0

      this.arquivo_vfc_atual = this.arquivos_vfc[0]
    }
  }

  select_vfc(arquivo_vfc:any){
    this.file = new File([''],this.arquivo_vfc_atual_index.toString())
    this.uploadForm.controls['valido'].setValue(true)
    this.uploadForm.controls['dados_input'].setValue(this.arquivo_vfc_atual )
    this.mostrar_vfc=!this.mostrar_vfc

  }

   downloadDataAsTxt(url: string, index:number) {
    console.log(url)
    this.cnnwebapiservice.downloadDataAsTxt(url).subscribe(
      data =>  this.arquivos_vfc[index]=data,
      error => console.log(error)
    );

  }

}
