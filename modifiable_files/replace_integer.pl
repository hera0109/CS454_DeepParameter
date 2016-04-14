#!/usr/bin/env perl
use strict;
use warnings;

# This perl script is used by 'expose_integers.bsh' to replace parameters. Of little use by itself
# Perl has to be used as it has a special form of regex which permits backwards and fowards references (not used in sed, the logical alternative)

my $num_args = $#ARGV +1;
if($num_args != 3) {
	print "\nUsage: replace_integer.pl occurance replacement to_replace";
	print "\nE.g. `./replace_integer.pl 1 \"INTEGER_PLACEHOLDER\" \"method(0 , 1, 6);\"`";
	print "\nWould output: \"method(INTEGER_PLACEMENT , 1, 6);\"\n";
	exit;
}

my $string =$ARGV[2];

my $cont =0;
sub replacen { 
        my ($index,$original,$replacement) = @_;
        $cont++;
        return $cont == $index ? $replacement: $original;
}

sub replace_quoted {
        my ($string, $index,$replacement) = @_;
        $cont = 0; # initialize match counter
        $string =~ s/((?<=(=|>|<|\+|-|\/|\*|\[|,|\(|\s))(?<!(e-))[0-9]+(?!([a-zA-Z0-9]|\.)))/replacen($index,$1,$replacement)/eg;
        return $string;
}

my $result = replace_quoted ( $string, $ARGV[0] ,$ARGV[1]);
print "$result\n";
