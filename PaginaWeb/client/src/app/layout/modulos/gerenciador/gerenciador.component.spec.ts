import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import {GerenciadorComponent } from './gerenciador.component';


describe('RepositorioComponent', () => {
  let component: GerenciadorComponent;
  let fixture: ComponentFixture<GerenciadorComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ GerenciadorComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(GerenciadorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
