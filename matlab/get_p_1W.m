function[v,E,E_rat,psi,dens,pt1]=get_p_1W(d1,w1)

bg=d1;
beta=2;
pot_heights=[bg,bg-d1,bg];
% find the minimum debroglie wavelength:
lambda=((2*pi)/sqrt(beta))*(1/sqrt(d1));
dx=(1/2)*(lambda/(2*pi));
n_t1=ceil(w1/dx);
w_bg=ceil(10*lambda);
n_bg=ceil(w_bg/dx);

pot_widths=[w_bg,w1,w_bg];
w_tot=sum(pot_widths);
n_steps=n_t1+2*n_bg;
%{
%pick the number of elements you want...
n_steps=1e2;
%}
x=linspace(0,w_tot,n_steps);
d_x=x(2)-x(1);
v=zeros(1,n_steps);

%build your v
bounds=zeros(length(pot_widths)+1,1);
bounds(end)=w_tot;

for j = 2:length(pot_widths)+1
    bounds(j)=sum(pot_widths(1:j-1));
end

for i = 1:n_steps
    
    if x(i)<bounds(2)
        v(i)=pot_heights(1);
    elseif x(i)<sum(bounds(3))
        v(i)=pot_heights(2);
    else
        v(i)=pot_heights(3);
    end   
    
end
%}

%create the components of the numerov matrix method
V=diag(v,0);
A=(-1/beta)*(1/dx^(2))*(diag(-2*ones(n_steps,1),0)+...
    diag(ones(n_steps-1,1),-1)+diag(ones(n_steps-1,1),+1))
B=(1/12)*(diag(10*ones(n_steps,1),0)+diag(ones(n_steps-1,1),-1)+...
    diag(ones(n_steps-1,1),+1))
sys=B\A+V;

% solve the numerov matrix method
[psi,E]=eig(sys);
[E,ind]=sort(diag(E));
psi=psi(:,ind);

% take out only the valid values of E.
inds=find(E<bg);
psi=psi(:,inds);
E=E(inds);
E_rat=(bg-E)/bg;

% get and normalize the densities:
w1b=min(find(v==bg-d1));
w1e=max(find(v==bg-d1));
dens=psi.*psi;
for i = 1:size(dens,2)
    d=dens(:,i);
    int=trapz(d);
    dens(:,i)=d/int;
    pt1(i)=trapz(d(w1b:w1e));
end
return
