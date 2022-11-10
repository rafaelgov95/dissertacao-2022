/**
 * Enum que define o tipo de uso do telefone.
 */
export enum UsoTelefone {
  Pessoal, Trabalho, Ramal, Recado
}

/**
 * Classe que define os atributos de um objeto Telefone.
 */
export class Telefone {

  // Quando nulo, este telefone não existe na base de dados.
  private _id: number;

  // Tipo de uso deste telefone.
  private _uso: UsoTelefone;

  // Caso seja um telefone para recados, qual o nome do contato.
  private _contato: string;

  // Número do telefone.
  private _numero: string;

  // Ramal (apenas em caso de telefone fixo)
  private _ramal: string;

  // Operadora (apenas em caso de celular)
  private _operadora: string;

  // Verdadeira caso tenha WhatsApp.
  private _isWhatsApp: boolean;

  // Ordem de prioridade deste telefone.
  private _prioridade: number;

  public isCelular(): boolean {
    return this.numero[2] == '9';
  }

  public isValido(): boolean {
    return true;
  }

  get id(): number {
    return this._id;
  }

  set id(id: number) {
    this._id = id;
  }

  get uso(): UsoTelefone {
    return this._uso;
  }

  set uso(uso: UsoTelefone) {
    this._uso = uso;
  }

  get contato(): string {
    return this.uso.toString() === 'Recado' ? this._contato : null;
  }

  set contato(contato: string) {
    this._contato = contato;
  }

  get numero(): string {
    return this._numero;
  }

  set numero(numero: string) {
    this._numero = numero;
  }

  get ramal(): string {
    return !this.isCelular() ? this._ramal : null;
  }

  set ramal(ramal: string) {
    this._ramal = ramal;
  }

  get operadora(): string {
    return this.isCelular() ? this._operadora : null;
  }

  set operadora(operadora: string) {
    this._operadora = operadora;
  }

  get isWhatsApp(): boolean {
    return this.isCelular() ? this._isWhatsApp : false;
  }

  set isWhatsApp(isWhatsApp: boolean) {
    this._isWhatsApp = isWhatsApp;
  }

  get prioridade(): number {
    return this._prioridade;
  }

  set prioridade(prioridade: number) {
    this._prioridade = prioridade;
  }
}
