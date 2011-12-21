#include <stdlib.h>
#include <stdio.h>
#define ARMA_NO_DEBUG
#include <armadillo>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_bessel.h>
#include <dispatch/dispatch.h>
//#include <complex.h>
using namespace std;
using namespace arma;


struct f_params { double mm; double d; };
struct f1_params { double mm; double d; double s; };
struct rparams { double v; double ss; };
struct mu1_params {  double v; double s;  vec vd; vec muy;};
struct initcond_params {double s; double x;}; 

double const pi=4.0*atan(1.0);
int const npoint=2000;
int const xpoints=1024*2*2*2*2;
int const xpoints2=4*1024;
int const nk=512*2*2;///2;
double const dz0=25.0/(nk-1);

static vec S(xpoints);
static vec vk(nk);
static vec dvk(nk-1);
//static vec vd;

static void
loadvk () {
	int j;
	for (j=0; j<nk; j++) {
		vk(j)=exp(j*dz0-10.0)*exp(j*dz0-10.0);
	};
	
    vec x0=vk;
	vec x1=vk;
	x0.shed_row(0);
	x1.shed_row(nk-1);
    
    dvk=x0-x1;
};


double invVdVfun(double s,double v) {
	
    double j0=gsl_sf_bessel_J0(sqrt(2.0)*s*sqrt(v));
    double j1=gsl_sf_bessel_J1(sqrt(2.0)*s*sqrt(v));
    double y0=gsl_sf_bessel_Y0(sqrt(2.0)*s*sqrt(v));
    double y1=gsl_sf_bessel_Y1(sqrt(2.0)*s*sqrt(v));
    
    double f1;
	double f2;
    double f3;
    
    f3=0.5*pi*y0/j0-M_EULER-log(s*sqrt(v)/sqrt(2.0));
    f1= 2.0*s*f3*f3; 
    f2= -0.5/v+ 0.5*pi*s/sqrt(2.0*v)*(j1*y0/(j0*j0)-y1/j0);
    return v*f2/f1;
}

double rangevfun(double v, void * p) {
	struct initcond_params * params 
	= (struct initcond_params *)p;
	double s = (params->s);
	double x = (params->x);
	double f1;
	
    f1=gsl_sf_bessel_J0(sqrt(2.0)*s*sqrt(v)) + 2.0*s*x*
    ((pi*gsl_sf_bessel_Y0(sqrt(2.0)*s*sqrt(v)))*0.5 - 
     gsl_sf_bessel_J0(sqrt(2.0)*s*sqrt(v))*(M_EULER + log((s*sqrt(v))/sqrt(2.0))));
    
    return f1;
}


double solveintervalend(double ss,double x)
{
    int status;
	int iter = 0, max_iter =1000;
	
	const gsl_root_fsolver_type *T;
	gsl_root_fsolver *s;
	double r;
    double x_lo;
	double x_hi;
	
    if (x<0){ 
        {x_lo=ss*0.000001;
            x_hi=10.0/(ss*ss);
        };
	}
    else {x_lo=ss*0.000001;
        x_hi=15.0/(ss*ss);
    };
    
	gsl_function F;
	struct initcond_params params = {ss,x};
	
	F.function = &rangevfun;
	F.params = &params;
	
	T = gsl_root_fsolver_brent;
	s = gsl_root_fsolver_alloc (T);
    
    
	gsl_root_fsolver_set (s, &F, x_lo, x_hi);
    
	do
	{
		iter++;
		status = gsl_root_fsolver_iterate (s);
		r = gsl_root_fsolver_root (s);
		x_lo = gsl_root_fsolver_x_lower (s);
		x_hi = gsl_root_fsolver_x_upper (s);
		status = gsl_root_test_interval (x_lo, x_hi,
										 0, 1.0e-15);
		
		
		
	}
	
	
	while (status == GSL_CONTINUE && iter < max_iter);
	
	double sol=gsl_root_fsolver_root (s);
	
	
	gsl_root_fsolver_free (s);
	
    //cout << sol << endl;
    return sol;
}



double
inttrapz(vec vecx,vec vecf)
{   
	//int nel=vecx.n_elem;
	
    int nel=nk;
	
	//vec x0=vecx;
	//vec x1=vecx;
	//x0.shed_row(0);
	//x1.shed_row(nel-1);
	vec y0=vecf;
	y0.shed_row(0);
	vec y1=vecf;
	y1.shed_row(nel-1);
	
	//vec dx=x0-x1;
	
	vec sy=y1+y0;
	
	
	return 0.5*dot(dvk,sy);
}

double
inttrapzN(vec vecx,vec vecf, int n)
{   
	int nel=n;
	
	vec x0=vecx;
	vec x1=vecx;
	x0.shed_row(0);
	x1.shed_row(nel-1);
	vec y0=vecf;
	y0.shed_row(0);
	vec y1=vecf;
	y1.shed_row(nel-1);
	
	vec dx=x0-x1;
	
	vec sy=y1+y0;
	
	
	return 0.5*dot(dx,sy);
}


double fun2(double m, double d,double s, double v,double y,double z)
{
	double mm=m+4/(3*pi)*v*s*s*s;
	double f;
	double r=pi*(1+2*s*s*(z+(y+mm))+s*s*s*s*(z-(y+mm))*(z-(y+mm)));
	
	if (y*y+d*d<=0.0){ {f=0.0;}}
	else {
        
        if (y+mm<=0.0) {{f=0.0;};}
        else {
            f=d*v*s*s*s*sqrt(y+mm)/(r*sqrt(y*y+d*d));
        };}
	
	return f;
	
}



double fun1(double m, double d, double s, double v,double z, double muy)
{
	double mm=m+4/(3*pi)*v*s*s*s-muy;
	double f;
	if ((z-mm)*(z-mm)+d*d<=0.0) {
		{f=0.5*sqrt(z);};
	}
	else {
		f=0.5*sqrt(z)*(1.0-(z-mm)/sqrt((z-mm)*(z-mm)+d*d));
        
	}
    
	//cout<<f<<endl;
    
	return f;
    
	
}

double intfun1(double m, vec d, double s, double v,vec z, vec muy)
{
	int niter=d.n_elem;
	int j;
	vec fv(niter);
	for (j=0; j<niter; j++) {
		
		fv(j)=fun1(m, d(j), s, v, z(j),muy(j));
		
	};
	
	return inttrapz(z, fv);
	
}



double
funmu1 (double m, void *p)
{struct mu1_params *params 
	= (struct mu1_params *) p;
	
	double s = params-> s;
	double v = params->v;
	vec vd=params-> vd;
    vec muy=params-> muy;
	
	return 1.0-1.5*intfun1(m, vd, s, v, vk, muy);
    
}


double fun2p(double m, double d,double s, double v,double y,double z)
{
	double mm=m+4.0/(3.0*pi)*v*s*s*s;
	double f;
	double r=pi*(1.0+2.0*s*s*(z+y)+s*s*s*s*(z-y)*(z-y));
	
	if (y*y-2.0*mm*y+mm*mm+d*d<=0.0) {{f=0.0;};}
	else {
		f=d*v*s*s*s*sqrt(y)/(r*sqrt((y-mm)*(y-mm)+d*d));
	}
	
	return f;
	
}

double intfun2vec(double m, vec dd,double s, double v,double z,vec muy)
{   double result;
	
	int j;
	vec gv(nk);
	
	for (j=0;j<nk; j++) {
		gv(j)=fun2p(m-muy(j),dd(j),s,v,vk(j),z);
	};
	
	result=inttrapz(vk,gv);
	
	return result;
	
	
}

vec fundeltaN(double m,vec dd,double s, double v,vec muy)
{
	vec dsol(nk);
	int j;
	for (j=0; j<nk; j++) {
		dsol(j)=intfun2vec(m,dd,s,v,vk(j),muy);
	}
    
	return dsol;
}



field<vec> solsbcsN (double ss, double v,double mu0,double m,vec vd,vec muy)
{
	
	int status;
	int iter = 0, max_iter =100;
	
	const gsl_root_fsolver_type *T;
	gsl_root_fsolver *s;
	double r = mu0;
	
	double x_lo;
	double x_hi;
	double mux=mu0;
	
	if (mu0<0) {{
		x_lo = -80.0*exp(1.5*sqrt(sqrt(sqrt(abs(mux))))+60.0)-10.0;
		//mu0*mu0*mu0-mu0*mu0*2+mu0-10;
		x_hi = -mu0+15.0;
		
	}
	}
	else {
		x_lo = mux*0.25*mux-200.0*(1.0-mux)-10.0;
		x_hi = 1.5*sqrt(abs(mux))+10.5;
		//mux*1.5*(1+exp(-2*mux*mux));
		
		
	}
	
	
	gsl_function F;
	struct mu1_params params = {v,ss,vd,muy};
	
	F.function = &funmu1;
	F.params = &params;
	
	T = gsl_root_fsolver_brent;
	s = gsl_root_fsolver_alloc (T);
	/*
     cout <<mu0<< endl;
     vec f_lo(2);
     f_lo(0)=x_lo;
     f_lo(1)=funmu1(x_lo, &params);
     
     vec f_hi(2);
     f_hi(0)=x_hi;
     f_hi(1)=funmu1(x_hi, &params);
     
     cout <<v<< endl;
     //cout << vd.max()<<endl;
     cout <<f_lo<< endl;
     cout <<f_hi<< endl;
    */
	
	gsl_root_fsolver_set (s, &F, x_lo, x_hi);
	//cout <<x_lo << endl;
	//cout <<x_hi << endl;
	
	
	
	do
	{
		iter++;
		status = gsl_root_fsolver_iterate (s);
		r = gsl_root_fsolver_root (s);
		x_lo = gsl_root_fsolver_x_lower (s);
		x_hi = gsl_root_fsolver_x_upper (s);
		status = gsl_root_test_interval (x_lo, x_hi,
										 0, 1.0e-4);
		
		
        //cout <<iter << endl;
    	
	}
	
	
	while (status == GSL_CONTINUE && iter < max_iter);
	 
    //vec vs;
    //vs<<iter<<v;
    //cout << vs << endl;

	//cout <<iter << endl;
	double musol=gsl_root_fsolver_root (s);
	
	
	gsl_root_fsolver_free (s);
	
	field<vec> solitos(3);	
	
	solitos(0)=iter;
	solitos(1)=vd;
	solitos(2)=musol;
	
	//cout <<musol<< endl;
	return solitos;
}


field<vec> itN1 (field<vec> GG, double ss, double vv, double points, vec muy ){
    
    double tmp1=GG(0)(0);
	vec    tmp2=GG(1);
	double tmp6=GG(2)(0);
	double muN;
	int iter;
	field<vec> res;
	
	vec vd=fundeltaN(tmp1,tmp2,ss,vv,muy);
	
	res=solsbcsN(ss,vv,tmp6,tmp1,vd,muy);
	
	muN=res(2)(0);
	iter=res(0)(0);
	
	field<vec> res1(3);
	
	res1(0)=muN;
	res1(1)=vd;
	res1(2)=iter;
	return res1;
}

double FockInt(double ss, double v, double m, double d, double y, double z, double muy){
    
    double mm=m+8.0/(3.0*pi)*v*ss*ss*ss-muy;
    double r=(1.0+2.0*ss*ss*(z+y)+ss*ss*ss*ss*(z-y)*(z-y));
    double vv=v*ss*ss*ss/pi;
	double f;

	if ((z-mm)*(z-mm)+d*d<=0.0) {
		{f=sqrt(z);};
	}
	else {
		f=sqrt(z)*(1.0-(z-mm)/sqrt((z-mm)*(z-mm)+d*d));
        
	}
        
    return  f*vv/r;
    
}

double Fockcont(double ss, double v, double m, vec vd, double y, vec muy){
    
    vec fi(nk);
    int i;
    for (i=0; i<nk; i++) {
        fi(i)=FockInt(ss, v, m, vd(i), y, vk(i), muy(i));
    };
    
    return -4.0/(3.0*pi)*v*ss*ss*ss+inttrapz(vk, fi);
}


vec SolFock(double ss, double v, double m, vec vd, vec muy0){
    
    vec muyN=muy0;
    vec muyN0(nk);
    double r=1;
    int i;
    int n=0;
    
    while (r>1.0e-4 && n< 50) {
    
        n++;    
    muyN0=muyN;
    for (i=0; i<nk; i++){
        muyN(i)=Fockcont(ss, v, m, vd, vk(i), muyN0);
    };
    r=norm(muyN-muyN0,2);
       // cout << n << endl;
}
    return muyN;
}
double preintOmega0(double v,double s, double m, double d, double x)
{
    double f;
    double ek=x-m -(4.0*s*s*s*v)/(3.0*pi);
    
    f=0.75*sqrt(x)*(x-m)*(1 - ek/sqrt(d*d + ek*ek));
    return f;
}


double preintOmegaU(double v,double s1, double m1, double m2, double d1, double d2,double x1,double x2){
    
    double f;
    double ek1=x1-m1 -(4.0*s1*s1*s1*v)/(3.0*pi);
    double ek2=x2-m2 -(4.0*s1*s1*s1*v)/(3.0*pi);
    // double inttheta=2/(1 +s1*s1*s1*s1*(x1 - x2)*(x1 - x2)+2*s1*s1*(x1 + x2));
    double inttheta=2.0/(-4.0*s1*s1*s1*s1*x1*x2 + (1.0 + s1*s1*(x1 + x2))*(1.0 + s1*s1*(x1 + x2)));
    
    f=-(v*s1*s1*s1*d1*d2)*inttheta*sqrt(x1*x2)/(sqrt(ek1*ek1 + d1*d1)*sqrt(ek2*ek2 + d2*d2));
    return f;
}

double preintOmegaF(double v,double s1, double m1, double m2, double d1, double d2,double x1,double x2){
    
    double f;
    double ek1=x1-m1 -(4.0*s1*s1*s1*v)/(3.0*pi);
    double ek2=x2-m2 -(4.0*s1*s1*s1*v)/(3.0*pi);
    // double inttheta=2/(1 +s1*s1*s1*s1*(x1 - x2)*(x1 - x2)+2*s1*s1*(x1 + x2));
    double inttheta=1.0/(-4.0*s1*s1*s1*s1*x1*x2 + (1.0 + s1*s1*(x1 + x2))*(1.0 + s1*s1*(x1 + x2)));
    
    f=v*s1*s1*s1*inttheta*sqrt(x1*x2)*(1.0-ek1/sqrt(ek1*ek1 + d1*d1))*(1.0-ek2/sqrt(ek2*ek2 + d2*d2));
    return 3.0*f/(8.0*pi);
}



field<vec> all_in_one(double ss, double eta, double sm, double sd){
     
    loadvk();
    
    double v=solveintervalend(ss,eta);
    
    //cout<< v <<endl;
    double mu0;
    double delta0;
    vec muy(nk);
    mu0=sm;
    delta0=sd;
    
    double mu1;
    vec delta1(nk); 
    mu1=mu0;
    delta1.ones();
    delta1=delta1*delta0;
    vec muy0(nk);
    
    muy.ones();
    muy=0.01*mu1*muy;
    
    muy=SolFock(ss,v, mu1, delta1,muy);
    
    //cout << muy.max() << endl;
    
    double tmp1;
	vec tmp2(nk);
	vec tmp3(nk);
	double errmu;
	double errdelta;
	double tmp4;
	double tmp5;

    
    tmp4=0;
    tmp5=0;
    tmp2=delta1;
    tmp1=mu1;
    
    field<vec> res2(3);
    
    vec vd=fundeltaN(tmp1,tmp2,ss,v,muy);
    res2=solsbcsN(ss,v,tmp1,tmp1,vd,muy);
    
    
    
    double muN=res2(2)(0);
    tmp3=abs(1.0-tmp2/(vd+1.0e-15));
    tmp4=tmp4+abs(1.0-tmp1/(muN+1.0e-15));
    
    if (tmp3.max()>tmp5){
        
        {tmp5=tmp3.max();}}
    else
    {tmp5=tmp5;}
    
    errmu=tmp4;
    errdelta=tmp5;
    
    vec DeltaN(nk);
    vec DeltaN1(nk);
    
    DeltaN=vd;
    DeltaN1=tmp2;
    muy=SolFock(ss,v, muN, DeltaN,muy);
    
    vec vdd;
	field<vec> GG(3);
	field<vec> vit(3);
    
    vec tmp0;
    
	int nn=0;
	while (errdelta>1.0e-4 && nn< 1000) {
             
            nn++;
            double tmp6=muN;
			tmp1=muN;
			tmp0=DeltaN1;
			tmp2=DeltaN;
			tmp5=0.0;
            tmp4=0.0;
        
			GG(0)=tmp1;
			GG(1)=tmp2;
			GG(2)=tmp6;
            
            muy=SolFock(ss,v, muN, DeltaN,muy);
            vit=itN1(GG, ss, v, nk,muy);
			
			vdd=vit(1);
			muN=vit(0)(0);
			
			
			DeltaN=abs(vdd);
			DeltaN1=abs(tmp2);
            
            tmp4=tmp4+abs(1.0-tmp1/(muN+1.0e-15));
            
            
            if (norm(vdd,2)<1.0e-15){
                
                {tmp3=0*tmp3;}}
            else
            {tmp3=abs(1.0-tmp2/(vdd));}
            
            if (tmp3.max()>tmp5){
                
                {tmp5=tmp3.max();}}
            else
            {tmp5=tmp5;}
            
            errmu=tmp4;
            errdelta=tmp5;
        //vec vs;
        //vs<< errdelta << v;
        
        }
	cout << nn << endl;
    cout << errdelta << endl;
    cout << errmu << endl;
    
    mat r(nk,nk);
    mat rF(nk,nk);
    vec rd0(nk);
    
    double Omega0;
    double OmegaU;
    double OmegaF;
    
    int i,j;
    
    for (i=0; i<nk; i++) 
    {
        for (j=0; j<nk; j++) 
        {
            r(i,j)=preintOmegaU(v,ss,muN-muy(i),muN-muy(j),DeltaN(i),DeltaN(j),vk(i),vk(j));
            rF(i,j)=preintOmegaF(v,ss,muN-muy(i),muN-muy(j),DeltaN(i),DeltaN(j),vk(i),vk(j));
            
        }
        rd0(i)=preintOmega0(v,ss,muN-muy(i),DeltaN(i),vk(i));
    };
    
    
    Omega0=inttrapzN(vk,rd0,nk);
    
    vec d1i(nk);
    vec rd;
    vec d1iF(nk);
    vec rdF;
    
    
    for (i=0; i<nk; i++)
    {
        rd=r.col(i);    
        d1i(i)=inttrapzN(vk,rd,nk);
        rdF=rF.col(i);    
        d1iF(i)=inttrapzN(vk,rdF,nk);
    }

    OmegaF=-(2.0*ss*ss*ss*v)/(3.0*pi)+inttrapzN(vk,d1iF,nk);

    OmegaU=3.0/(16.0*pi)*inttrapzN(vk,d1i,nk)-(2.0*ss*ss*ss*v)/(3.0*pi)+OmegaF;
    
    
    double Contact;
    
    Contact=-OmegaU/invVdVfun(ss,v); 
    
    field<vec> result(8);
    result(0)=DeltaN1;
    result(1)=DeltaN;
    result(2)=muN;
    result(3)=muy;
    result(4)=Omega0;
    result(5)=OmegaU;
    result(6)=OmegaF;
    result(7)=Contact;
	
return result;


}

vec meshsigma(double ss0, int npower, int interval ){
    
    double x0=log(ss0);
    double xf=log(ss0*pow(10.0,npower));
    double stepx=(xf-x0)/(interval-1);
    vec ssvec(interval);                  
    int i;              
    for (i=0;i<interval;i++){
                      ssvec(i)=exp(x0+stepx*i);
                  }
    return ssvec;
}

vec mesheta(double e0,double e1, int interval ){
    
    double stepe=(e1-e0)/(interval-1);
    vec evec(interval);                  
    int i;              
    for (i=0;i<interval;i++){
        evec(i)=e0+stepe*i;
    }
    return evec;
}


int
main (void){
	wall_clock tt;
	
    tt.tic();
	
    //double ss0=0.005;
    //int power=3;
    //int points=7;
    double eta0=-10.0;
    double etaf=20.0;
    int points_eta=1024;
    
    vec veta=mesheta(eta0,etaf, points_eta);
    //double eta=0.0;
    int i;
    vec vss;
    
    //vss=meshsigma(ss0, power,points);
    
    vss << 0.005 << 0.01 << 0.05 << 0.1 << 0.5 << 1.0 << 5.0;
   
    //vss << 0.1 ;
    
    // cout<< vss<< endl;
   // cout<< veta<< endl;
    
    int points_sigma=vss.n_elem;        
    
    
    int points=points_sigma*points_eta;
    
    //cout<< points<< endl;
  
   // cout<< solveintervalend(vss(0),veta(1))<<endl;

    
//    double eta=veta(0);
    
    double vparSigma[points];
    double vparEta[points];
    
    int j;
    
    for (i=0; i<points_eta;i++){
        for (j=0; j<points_sigma;j++){
            vparSigma[i+j*points_eta]=vss(j);
            vparEta[i+j*points_eta]=veta(i);
            
        }
    };
    
    
    //cout << vparS <<endl;
    
    double  vx1[points];
    double  vx2[points];
    double  vx3[points];
    double  vx4[points];
 
    double *x1 = vx1;
    double *x2 = vx2;
    double *x3 = vx3;
    double *x4 = vx4;
    
    double *xsigma = vparSigma;
    double *xeta = vparEta;
    
    dispatch_apply(points, dispatch_get_global_queue(0, 0), ^(size_t i){
        
        cout << i+1 <<endl;
        //cout<< xsigma[i]<<endl;
       // cout<< xeta[i]<<endl;
        
                
        field<vec> rr=all_in_one(xsigma[i], xeta[i], 0.9, 0.1);
        x1[i]=rr(1).max();
        x2[i]=rr(2)(0);
        x3[i]=rr(7)(0);
        
        vec pp=-(rr(4)+rr(5)); 
        
        x4[i]=pp(0);
        
        
        
    });
    
    vec y1(points);
    vec y2(points);   
    vec y3(points);
    vec y4(points);
 
    for(i=0; i<points; i++){
        y1(i)=x1[i];
        y2(i)=x2[i];
        y3(i)=x3[i];
        y4(i)=x4[i];
        
    }
    
    y1.save("etaF_dvec.dat", raw_ascii);
    y2.save("etaF_mvec.dat", raw_ascii);
    y3.save("etaF_cvec.dat", raw_ascii);
    y4.save("etaF_pvec.dat", raw_ascii);
    vss.save("etaF_svec.dat", raw_ascii);
    veta.save("etaF_evec.dat", raw_ascii);
    
    
    //cout<< y1<< endl;
    //cout<< y2<<endl;               
    //cout<< y3<<endl;
    //cout<< y4<<endl;
    
    
	double n_secs = tt.toc();
	cout << "took " << n_secs << " seconds" << endl;
	
	return 0;
}