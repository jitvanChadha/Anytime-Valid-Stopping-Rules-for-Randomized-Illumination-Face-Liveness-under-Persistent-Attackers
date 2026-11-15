use strict; use POSIX qw(erf ceil); 
sub Phi { 0.5*(1+erf($_[0]/sqrt(2))) }
sub zq { my $p=shift; my($a,$b)=(-10,10); for(1..80){my $m=($a+$b)/2; if(Phi($m)<$p){$a=$m}else{$b=$m}} ($a+$b)/2 }
sub binom { my($n,$k,$p)=@_; my $c=1; for my $i(1..$k){$c*=($n-$k+$i)/$i} $c*$p**$k*(1-$p)**($n-$k) }
my ($delta,$lazy,$pi,$alpha,$beta)=(50,150,0.1,1e-3,1e-2);
my $zs=zq(1-$alpha)+zq(1-$beta); printf "z_alpha+z_beta=%.4f sq=%.3f R=%.4f\n",$zs,$zs**2,2*log(1/$alpha)/$zs**2;
for my $s (10,11.756,15,20,25,35,50){
  my $M=ceil(($zs*$s/$delta)**2); my $sd=$s/sqrt($M);
  my $f=sub{my $c=shift; my $t=0; for my $k(0..$M){ my $w=binom($M,$k,1-$pi); my $m=($k*$lazy+($M-$k)*$delta)/$M; $t+=$w*Phi(($c-$m)/$sd)} $t-$alpha};
  my($a,$b)=($delta-8*$s,$lazy); for(1..80){my $m=($a+$b)/2; if($f->($m)<0){$a=$m}else{$b=$m}} my $c=($a+$b)/2;
  my $z=($c-$delta)/$sd; my $ap=$pi*Phi($z)+(1-$pi)*Phi(($c-$lazy)/$sd);
  printf "sigma=%6.3f M=%2d c=%7.3f z=(c-delta)sqrt(M)/sigma=%6.3f APCER_persist=%.5f  /alpha=%.2f\n",$s,$M,$c,$z,$ap,$ap/$alpha;
}
# overshoot MC for Gaussian walk N(a,2a), a=delta^2/(2 sigma^2), upper boundary only and with lower boundary beta
srand(7); sub gauss { my($u,$v)=(rand(),rand()); sqrt(-2*log($u+1e-300))*cos(2*3.141592653589793*$v) }
for my $s (25,11.756){ my $a=$delta**2/(2*$s*$s); my $sd=sqrt(2*$a); my $b=log(1/$alpha); my $lo=log($beta); my($n,$ST,$SR,$SR2)=(200000,0,0,0);
 for(1..$n){ my($S,$t)=(0,0); while(1){ $S+=$a+$sd*gauss(); $t++; if($S>=$b){ $SR+=$S-$b; last } if($S<=$lo){ last } if($t>=200){last} } $ST+=$t; }
 printf "sigma=%.3f a=%.3f  E[tau](live,c=0,two-sided,T<=200) = %.3f ; first-order b/a=%.3f; Wald approx=%.3f; mean overshoot(on accepts)~%.3f\n",$s,$a,$ST/$n,$b/$a,((1-$beta)*log((1-$beta)/$alpha)+$beta*log($beta/(1-$alpha)))/$a, $SR/($n*(1-$beta)) }
