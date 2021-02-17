function[v,E,t1_E,t2_E,t3_E,psi,dens,p_1,p_2,p_int,p_bg]=get_p_2W(d1,d2,w1,w2,w_sep)
w=[w1,w_sep,w2];

bg=max(d1,d2);
pot_heights=[bg,bg-d1,bg,bg-d2,bg];
beta=2;

% find the minimum debroglie wavelength:

dx1=1/sqrt(beta*bg);
dx2=nanmin(w)/5;
% note temporary difference: 020421
%dx=min(dx1,dx2)/100;
dx=min(dx1,dx2);

lambda=2*pi*dx1;

pw=w;
pw(isnan(pw))=0;
w1=pw(1);
w_sep=pw(2);
w2=pw(3);
n_t1=ceil(w1/dx);
n_t2=ceil(w2/dx);
n_sep=ceil(w_sep/dx);

%w_bg=ceil(2.5*lambda);
% temp change, 020121
w_bg=ceil(10*lambda);
n_bg=ceil(w_bg/dx);

pot_widths=[w_bg,pw,w_bg];
w_tot=sum(pot_widths);

n_steps=n_t1+n_t2+n_sep+2*n_bg;
x=linspace(0,w_tot,n_steps);
del_x=x(2)-x(1);
v=zeros(1,n_steps);
%build your v
bounds=zeros(length(pot_widths)+1,1);
bounds(end)=w_tot;

for j = 2:length(pot_widths)+1
    bounds(j)=sum(pot_widths(1:j-1));
end
bounds;

%background ends at n_bg+1, t1 begins at n_bg+2
t1b_int=n_bg+2;
t1e_int=t1b_int+n_t1-1;
t2b_int=t1e_int+n_sep;
t2e_int=t2b_int+n_t2-1;

for i = 1:n_steps
    
    if x(i)<=bounds(2)
        v(i)=pot_heights(1);
    elseif x(i)<=sum(bounds(3))
        
        v(i)=pot_heights(2);
    elseif x(i)<=sum(bounds(4))
        
        v(i)=pot_heights(3);
    elseif x(i)<=sum(bounds(5))
        
        v(i)=pot_heights(4);
    else
 
        v(i)=pot_heights(5);
    end   
    
end

%}
% solve the schrodinger equation using the numerov matrix method. 
V=diag(v,0);
A=(-1/beta)*(1/dx^(2))*(diag(-2*ones(n_steps,1),0)+diag(ones(n_steps-1,1),-1)+diag(ones(n_steps-1,1),+1));
B=(1/12)*(diag(10*ones(n_steps,1),0)+diag(ones(n_steps-1,1),-1)+diag(ones(n_steps-1,1),+1));
sys=B\A+V;
[psi,E]=eig(sys);
[E,inds]=sort(diag(E));
psi=psi(:,inds);
inds=find(and(E>0,E<bg));
E=E(inds);
psi=psi(:,inds);
dens=psi.*psi;

% normalizing the densities.
p_1=zeros(1,length(E));
p_2=zeros(1,length(E));
p_int=zeros(1,length(E));
p_bg=zeros(1,length(E));
E_dom=nan(3,length(E));
for i = 1:length(E)
    d=dens(:,i);
    int=trapz(d);
    d=d/int;
    dens(:,i)=d;
    
    if pw(1)==0
        p_1(i)=0;
    else
        p_1(i)=trapz(d(t1b_int:t1e_int));
    end

    if pw(3)==0
        p_2(i)=0;
    else
        p_2(i)=trapz(d(t2b_int:t2e_int));
    end
    
    if pw(2)==0
        p_int(i)=0;
    else
        p_int(i)=trapz(d(t1e_int+1:t2b_int-1));
    end
    p_bg(i)=1-sum([p_1(i),p_2(i),p_int(i)]);
    
    if and(p_1(i)>p_2(i),p_1(i)>p_bg(i))
        E_dom(1,i)=E(i);
    elseif and(p_2(i)>p_1(i),p_2(i)>p_bg(i))
        E_dom(2,i)=E(i);
    else
        E_dom(3,i) = E(i);
    end
    
end

t3_E = E_dom(3,~isnan(E_dom(3,:)));
t2_E=E_dom(2,~isnan(E_dom(2,:)));
t1_E=E_dom(1,~isnan(E_dom(1,:)));

%{
figure()
plot(x,(v./max(v))-.8)

%}
%{
hold on
plot(x,psi(:,19))
figure()
plot(t1_E)
hold on
plot(t2_E)
figure()
plot(E_dom','+')
%}

return