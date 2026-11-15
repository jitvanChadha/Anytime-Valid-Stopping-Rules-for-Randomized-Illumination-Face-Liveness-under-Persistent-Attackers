use strict; use List::Util qw(max);
my ($delta,$lazy,$pi)=(50,150,0.1);
sub drift { my $s=shift; my ($n,$sum)=(40000,0); my $lo=$delta-10*$s; my $hi=$delta+10*$s; my $h=($hi-$lo)/$n;
 for my $i (0..$n){ my $y=$lo+$i*$h; my $w=($i==0||$i==$n)?1:(($i%2)?4:2);
  my $f=exp(-($y-$delta)**2/(2*$s*$s))/($s*sqrt(2*3.141592653589793));
  my $a=-($y-$delta)**2/(2*$s*$s)+log($pi); my $b=-($y-$lazy)**2/(2*$s*$s)+log(1-$pi);
  my $m=max($a,$b); my $lse=$m+log(exp($a-$m)+exp($b-$m));
  my $ll=-($y**2)/(2*$s*$s)-$lse; $sum+=$w*$f*$ll; } return $sum*$h/3; }
for my $s (10,15,20,24,24.17,25,35,50,11.756){ printf "sigma=%6.3f delta/sigma=%.4f drift=%+.5f\n",$s,$delta/$s,drift($s); }
my ($a,$b)=(15,30); for (1..60){ my $m=($a+$b)/2; if(drift($m)<0){$a=$m}else{$b=$m} } printf "root sigma=%.4f ratio=%.4f\n",$a,$delta/$a;
