export class DataECG{

    public Erro : boolean
    public Arritmia: number
    public Hipertensao: number
    public Saudavel : number
    public Plot_Filtrada_x_axis:[]
    public Plot_Filtrada_y_axis:[]
    public Plot_Original_y_axis:[]
    public Plot_Original_x_axis:[]
    public Plot_Wavelet:[]
    constructor (){
        this.Plot_Original_x_axis=[]
        this.Plot_Original_x_axis=[]
        this.Erro=false
        this.Arritmia=0
        this.Hipertensao=0
        this.Saudavel=0
        this.Plot_Filtrada_x_axis=[]
        this.Plot_Filtrada_y_axis=[]
        this.Plot_Original_y_axis=[]
        this.Plot_Original_x_axis=[]
        this.Plot_Wavelet=[]
    }

}
