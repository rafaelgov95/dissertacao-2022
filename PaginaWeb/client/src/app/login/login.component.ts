import { User } from './../shared/modelos/user-modelo';
import { element } from 'protractor';
import { LoginService } from './../shared/services/login/login-service';

import { routerTransition } from '../router.animations';
import { Component, OnInit } from '@angular/core';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { Router, ActivatedRoute } from '@angular/router';
import { from, Observable, Subscription } from 'rxjs';
import { filter, map } from 'rxjs/operators';
import { fromEvent } from 'rxjs';
// import { ToastrService } from 'ngx-toastr';


@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss'],
  animations: [routerTransition()]

})
export class LoginComponent implements OnInit {
  isLoding = false

  emailRegex = '^[a-z0-9]+(\.[_a-z0-9]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,15})$';
  alert = true;

  user: User;
  submitted = false;
  users=[{nome:"rafael",senha:"12345"},{nome:"milton",senha:"ufms12345"},{nome:"felipe",senha:"ufms12345"},{nome:"renato",senha:"ufms@2020"},{nome:"cremildo",senha:"ufms@2020"},{nome:"teste",senha:"ufms@2021"}]

  constructor(
    // private toastr: ToastrService,
    private fb: FormBuilder,
    private route: ActivatedRoute,
    private router: Router,
    private loginService: LoginService,
    // private returnUrl: any,
    private UserForm: FormGroup,
    private subscription: Subscription

  ) {
    this.user = new User('', '');
  }

  ngOnInit(): void {
    this.subscription = fromEvent(document, 'keypress').subscribe(e => {
      // if (e['key'] == "Enter") {
      //   this.login()
      // }
    })
    this.buildForm();
    this.loginService.logout();
    // this.returnUrl = this.route.snapshot.queryParams['returnUrl'] || 'home';
  }

  buildForm(): void {
    this.UserForm = this.fb.group({
      'nome': [this.user.nome, [
        Validators.required,
        Validators.minLength(4)
      ]
      ],
      'senha': [this.user.senha, [
        Validators.required,
        Validators.minLength(4)
      ]]
    });

    this.UserForm.valueChanges
      .subscribe(data => this.onValueChanged(data));

    this.onValueChanged();
  }

  onValueChanged(data?: any) {
    if (!this.UserForm) { return; }
    const form = this.UserForm;

    for (const field in this.formErrors) {
      // clear previous error message (if any)
      // this.formErrors[field] = '';
      const control = form.get(field);

      // if (control && control.dirty && !control.valid) {
      //   const messages = this.validationMessages[field];
      //   for (const key in control.errors) {
      //     this.formErrors[field] += messages[key] + ' ';
      //   }
      // }
    }
  }

  formErrors = {
    'nome': ''
    ,
    'senha': ''
  };

  validationMessages = {
    'nome': {
      'required': 'Nome de usuário requerido.',
      'minlength': 'Nome tem que possuir mais de 4 caracteres'
    },
    'senha': {
      'required': 'Senha requirida.',
      'minlength': 'Senha tem que possuir mais de 4 caracteres'
    }
  };

  Lodingspinner() {
    this.isLoding = !this.isLoding
  }

  login() {
      let usuario = this.users.filter(x=>x.nome==this.UserForm.controls['nome'].value && x.senha==this.UserForm.controls['senha'].value)
      console.log(usuario)
      if(usuario.length>0){
          localStorage.setItem('currentUser', JSON.stringify(usuario));
          this.router.navigate(['/home']);
      }else{
        this.Error()
      }

  }
  Error() {
    // this.toastr.error('Login Invalído', 'Tente novamente!');
  }
}
