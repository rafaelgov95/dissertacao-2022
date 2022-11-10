
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';
import { Router, CanActivate } from '@angular/router';
import { LoginService } from '../services/login/login-service';

@Injectable()
export class AuthGuard implements CanActivate {

    constructor(private servico: LoginService, private router: Router) { }

    canActivate(){
        if(this.servico.isLoggedIn())
            return true
        this.router.navigate(['/login']);
        return false
    }
}
