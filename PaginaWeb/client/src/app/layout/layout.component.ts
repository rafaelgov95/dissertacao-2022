import { Component, OnInit } from '@angular/core';
import { Router, NavigationStart, NavigationEnd, NavigationCancel, NavigationError } from '@angular/router';
// import { ToastrService } from 'ngx-toastr';

@Component({
    selector: 'app-layout',
    templateUrl: './layout.component.html',
    styleUrls: ['./layout.component.scss']
})
export class LayoutComponent implements OnInit {
    public routeLoading: boolean = false;

    constructor(public router: Router) {
        this.router.events.subscribe((event) => {
            if (event instanceof NavigationStart) {
              this.routeLoading = true;
            }

            if (event instanceof NavigationEnd ||
              event instanceof NavigationCancel ||
              event instanceof NavigationError)
            {
                this.routeLoading = false;
            }
          });
    }

    ngOnInit() {
        if (this.router.url === '/') {
            this.router.navigate(['/home']);
        }
    }

}
