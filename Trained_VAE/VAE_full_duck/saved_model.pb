®¼
¢ø
:
Add
x"T
y"T
z"T"
Ttype:
2	
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
3
Square
x"T
y"T"
Ttype:
2
	
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-0-ga4dfb8d1a718´	
±
*autoencoder/encoder_PI/dense_100_PI/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Äd*;
shared_name,*autoencoder/encoder_PI/dense_100_PI/kernel
ª
>autoencoder/encoder_PI/dense_100_PI/kernel/Read/ReadVariableOpReadVariableOp*autoencoder/encoder_PI/dense_100_PI/kernel*
_output_shapes
:	Äd*
dtype0
¨
(autoencoder/encoder_PI/dense_100_PI/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*9
shared_name*(autoencoder/encoder_PI/dense_100_PI/bias
¡
<autoencoder/encoder_PI/dense_100_PI/bias/Read/ReadVariableOpReadVariableOp(autoencoder/encoder_PI/dense_100_PI/bias*
_output_shapes
:d*
dtype0
ª
'autoencoder/encoder_PI/dense_284/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d2*8
shared_name)'autoencoder/encoder_PI/dense_284/kernel
£
;autoencoder/encoder_PI/dense_284/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/encoder_PI/dense_284/kernel*
_output_shapes

:d2*
dtype0
¢
%autoencoder/encoder_PI/dense_284/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%autoencoder/encoder_PI/dense_284/bias

9autoencoder/encoder_PI/dense_284/bias/Read/ReadVariableOpReadVariableOp%autoencoder/encoder_PI/dense_284/bias*
_output_shapes
:2*
dtype0
ª
'autoencoder/encoder_PI/dense_285/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*8
shared_name)'autoencoder/encoder_PI/dense_285/kernel
£
;autoencoder/encoder_PI/dense_285/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/encoder_PI/dense_285/kernel*
_output_shapes

:2*
dtype0
¢
%autoencoder/encoder_PI/dense_285/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%autoencoder/encoder_PI/dense_285/bias

9autoencoder/encoder_PI/dense_285/bias/Read/ReadVariableOpReadVariableOp%autoencoder/encoder_PI/dense_285/bias*
_output_shapes
:*
dtype0
ª
'autoencoder/encoder_PI/dense_286/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*8
shared_name)'autoencoder/encoder_PI/dense_286/kernel
£
;autoencoder/encoder_PI/dense_286/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/encoder_PI/dense_286/kernel*
_output_shapes

:2*
dtype0
¢
%autoencoder/encoder_PI/dense_286/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%autoencoder/encoder_PI/dense_286/bias

9autoencoder/encoder_PI/dense_286/bias/Read/ReadVariableOpReadVariableOp%autoencoder/encoder_PI/dense_286/bias*
_output_shapes
:*
dtype0
ª
'autoencoder/decoder_PI/dense_287/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*8
shared_name)'autoencoder/decoder_PI/dense_287/kernel
£
;autoencoder/decoder_PI/dense_287/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/decoder_PI/dense_287/kernel*
_output_shapes

:d*
dtype0
¢
%autoencoder/decoder_PI/dense_287/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*6
shared_name'%autoencoder/decoder_PI/dense_287/bias

9autoencoder/decoder_PI/dense_287/bias/Read/ReadVariableOpReadVariableOp%autoencoder/decoder_PI/dense_287/bias*
_output_shapes
:d*
dtype0
«
'autoencoder/decoder_PI/dense_288/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	dÄ*8
shared_name)'autoencoder/decoder_PI/dense_288/kernel
¤
;autoencoder/decoder_PI/dense_288/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/decoder_PI/dense_288/kernel*
_output_shapes
:	dÄ*
dtype0
£
%autoencoder/decoder_PI/dense_288/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ä*6
shared_name'%autoencoder/decoder_PI/dense_288/bias

9autoencoder/decoder_PI/dense_288/bias/Read/ReadVariableOpReadVariableOp%autoencoder/decoder_PI/dense_288/bias*
_output_shapes	
:Ä*
dtype0
ª
'autoencoder/decoder_PC/dense_289/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*8
shared_name)'autoencoder/decoder_PC/dense_289/kernel
£
;autoencoder/decoder_PC/dense_289/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/decoder_PC/dense_289/kernel*
_output_shapes

:2*
dtype0
¢
%autoencoder/decoder_PC/dense_289/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*6
shared_name'%autoencoder/decoder_PC/dense_289/bias

9autoencoder/decoder_PC/dense_289/bias/Read/ReadVariableOpReadVariableOp%autoencoder/decoder_PC/dense_289/bias*
_output_shapes
:2*
dtype0
ª
'autoencoder/decoder_PC/dense_290/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*8
shared_name)'autoencoder/decoder_PC/dense_290/kernel
£
;autoencoder/decoder_PC/dense_290/kernel/Read/ReadVariableOpReadVariableOp'autoencoder/decoder_PC/dense_290/kernel*
_output_shapes

:2*
dtype0
¢
%autoencoder/decoder_PC/dense_290/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%autoencoder/decoder_PC/dense_290/bias

9autoencoder/decoder_PC/dense_290/bias/Read/ReadVariableOpReadVariableOp%autoencoder/decoder_PC/dense_290/bias*
_output_shapes
:*
dtype0
¾
0autoencoder/encoder_image/dense_100_image/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*A
shared_name20autoencoder/encoder_image/dense_100_image/kernel
·
Dautoencoder/encoder_image/dense_100_image/kernel/Read/ReadVariableOpReadVariableOp0autoencoder/encoder_image/dense_100_image/kernel* 
_output_shapes
:
d*
dtype0
´
.autoencoder/encoder_image/dense_100_image/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*?
shared_name0.autoencoder/encoder_image/dense_100_image/bias
­
Bautoencoder/encoder_image/dense_100_image/bias/Read/ReadVariableOpReadVariableOp.autoencoder/encoder_image/dense_100_image/bias*
_output_shapes
:d*
dtype0
°
*autoencoder/encoder_image/dense_291/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*;
shared_name,*autoencoder/encoder_image/dense_291/kernel
©
>autoencoder/encoder_image/dense_291/kernel/Read/ReadVariableOpReadVariableOp*autoencoder/encoder_image/dense_291/kernel*
_output_shapes

:d*
dtype0
¨
(autoencoder/encoder_image/dense_291/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(autoencoder/encoder_image/dense_291/bias
¡
<autoencoder/encoder_image/dense_291/bias/Read/ReadVariableOpReadVariableOp(autoencoder/encoder_image/dense_291/bias*
_output_shapes
:*
dtype0
°
*autoencoder/encoder_image/dense_292/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*;
shared_name,*autoencoder/encoder_image/dense_292/kernel
©
>autoencoder/encoder_image/dense_292/kernel/Read/ReadVariableOpReadVariableOp*autoencoder/encoder_image/dense_292/kernel*
_output_shapes

:d*
dtype0
¨
(autoencoder/encoder_image/dense_292/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(autoencoder/encoder_image/dense_292/bias
¡
<autoencoder/encoder_image/dense_292/bias/Read/ReadVariableOpReadVariableOp(autoencoder/encoder_image/dense_292/bias*
_output_shapes
:*
dtype0
°
*autoencoder/decoder_image/dense_293/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*;
shared_name,*autoencoder/decoder_image/dense_293/kernel
©
>autoencoder/decoder_image/dense_293/kernel/Read/ReadVariableOpReadVariableOp*autoencoder/decoder_image/dense_293/kernel*
_output_shapes

:2*
dtype0
¨
(autoencoder/decoder_image/dense_293/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*9
shared_name*(autoencoder/decoder_image/dense_293/bias
¡
<autoencoder/decoder_image/dense_293/bias/Read/ReadVariableOpReadVariableOp(autoencoder/decoder_image/dense_293/bias*
_output_shapes
:2*
dtype0
°
*autoencoder/decoder_image/dense_294/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2d*;
shared_name,*autoencoder/decoder_image/dense_294/kernel
©
>autoencoder/decoder_image/dense_294/kernel/Read/ReadVariableOpReadVariableOp*autoencoder/decoder_image/dense_294/kernel*
_output_shapes

:2d*
dtype0
¨
(autoencoder/decoder_image/dense_294/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*9
shared_name*(autoencoder/decoder_image/dense_294/bias
¡
<autoencoder/decoder_image/dense_294/bias/Read/ReadVariableOpReadVariableOp(autoencoder/decoder_image/dense_294/bias*
_output_shapes
:d*
dtype0
²
*autoencoder/decoder_image/dense_295/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
d*;
shared_name,*autoencoder/decoder_image/dense_295/kernel
«
>autoencoder/decoder_image/dense_295/kernel/Read/ReadVariableOpReadVariableOp*autoencoder/decoder_image/dense_295/kernel* 
_output_shapes
:
d*
dtype0
ª
(autoencoder/decoder_image/dense_295/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(autoencoder/decoder_image/dense_295/bias
£
<autoencoder/decoder_image/dense_295/bias/Read/ReadVariableOpReadVariableOp(autoencoder/decoder_image/dense_295/bias*
_output_shapes

:*
dtype0

NoOpNoOp
¥W
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*àV
valueÖVBÓV BÌV
Ê

encoder_PI

decoder_PI
sampling
shared_decoder
encoder_image
decoder_image
regularization_losses
trainable_variables
		variables

	keras_api

signatures

	dense_100
dense_50

dense_mean
	dense_var
regularization_losses
trainable_variables
	variables
	keras_api
s
	dense_100
dense_output
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
r
dense_20
dense_output
 regularization_losses
!trainable_variables
"	variables
#	keras_api

$	dense_100
%
dense_mean
&	dense_var
'regularization_losses
(trainable_variables
)	variables
*	keras_api

+dense_50
,	dense_100
-dense_output
.regularization_losses
/trainable_variables
0	variables
1	keras_api
 
Ö
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27
Ö
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27
­
regularization_losses
trainable_variables
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
		variables
 
h

2kernel
3bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
h

4kernel
5bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

6kernel
7bias
[regularization_losses
\trainable_variables
]	variables
^	keras_api
h

8kernel
9bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
 
8
20
31
42
53
64
75
86
97
8
20
31
42
53
64
75
86
97
­
regularization_losses
trainable_variables
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
h

:kernel
;bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
h

<kernel
=bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
 

:0
;1
<2
=3

:0
;1
<2
=3
­
regularization_losses
trainable_variables
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
 
 
 
­
regularization_losses
trainable_variables
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
h

>kernel
?bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
j

@kernel
Abias
~regularization_losses
trainable_variables
	variables
	keras_api
 

>0
?1
@2
A3

>0
?1
@2
A3
²
 regularization_losses
!trainable_variables
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
"	variables
l

Bkernel
Cbias
regularization_losses
trainable_variables
	variables
	keras_api
l

Dkernel
Ebias
regularization_losses
trainable_variables
	variables
	keras_api
l

Fkernel
Gbias
regularization_losses
trainable_variables
	variables
	keras_api
 
*
B0
C1
D2
E3
F4
G5
*
B0
C1
D2
E3
F4
G5
²
'regularization_losses
(trainable_variables
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
l

Hkernel
Ibias
regularization_losses
trainable_variables
	variables
	keras_api
l

Jkernel
Kbias
regularization_losses
trainable_variables
	variables
	keras_api
l

Lkernel
Mbias
 regularization_losses
¡trainable_variables
¢	variables
£	keras_api
 
*
H0
I1
J2
K3
L4
M5
*
H0
I1
J2
K3
L4
M5
²
.regularization_losses
/trainable_variables
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
0	variables
pn
VARIABLE_VALUE*autoencoder/encoder_PI/dense_100_PI/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE(autoencoder/encoder_PI/dense_100_PI/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'autoencoder/encoder_PI/dense_284/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%autoencoder/encoder_PI/dense_284/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'autoencoder/encoder_PI/dense_285/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%autoencoder/encoder_PI/dense_285/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'autoencoder/encoder_PI/dense_286/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%autoencoder/encoder_PI/dense_286/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'autoencoder/decoder_PI/dense_287/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%autoencoder/decoder_PI/dense_287/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'autoencoder/decoder_PI/dense_288/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%autoencoder/decoder_PI/dense_288/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'autoencoder/decoder_PC/dense_289/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%autoencoder/decoder_PC/dense_289/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'autoencoder/decoder_PC/dense_290/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%autoencoder/decoder_PC/dense_290/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0autoencoder/encoder_image/dense_100_image/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.autoencoder/encoder_image/dense_100_image/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*autoencoder/encoder_image/dense_291/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(autoencoder/encoder_image/dense_291/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*autoencoder/encoder_image/dense_292/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(autoencoder/encoder_image/dense_292/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*autoencoder/decoder_image/dense_293/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(autoencoder/decoder_image/dense_293/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*autoencoder/decoder_image/dense_294/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(autoencoder/decoder_image/dense_294/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*autoencoder/decoder_image/dense_295/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(autoencoder/decoder_image/dense_295/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
 
*
0
1
2
3
4
5
 
 
 
 

20
31

20
31
²
Sregularization_losses
Ttrainable_variables
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
U	variables
 

40
51

40
51
²
Wregularization_losses
Xtrainable_variables
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
Y	variables
 

60
71

60
71
²
[regularization_losses
\trainable_variables
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
]	variables
 

80
91

80
91
²
_regularization_losses
`trainable_variables
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
a	variables
 

0
1
2
3
 
 
 
 

:0
;1

:0
;1
²
hregularization_losses
itrainable_variables
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
j	variables
 

<0
=1

<0
=1
²
lregularization_losses
mtrainable_variables
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
n	variables
 

0
1
 
 
 
 
 
 
 
 
 

>0
?1

>0
?1
²
zregularization_losses
{trainable_variables
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
|	variables
 

@0
A1

@0
A1
³
~regularization_losses
trainable_variables
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
 

0
1
 
 
 
 

B0
C1

B0
C1
µ
regularization_losses
trainable_variables
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
	variables
 

D0
E1

D0
E1
µ
regularization_losses
trainable_variables
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
	variables
 

F0
G1

F0
G1
µ
regularization_losses
trainable_variables
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
	variables
 

$0
%1
&2
 
 
 
 

H0
I1

H0
I1
µ
regularization_losses
trainable_variables
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
	variables
 

J0
K1

J0
K1
µ
regularization_losses
trainable_variables
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
	variables
 

L0
M1

L0
M1
µ
 regularization_losses
¡trainable_variables
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
¢	variables
 

+0
,1
-2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿÄ
~
serving_default_input_2Placeholder*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
·
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2*autoencoder/encoder_PI/dense_100_PI/kernel(autoencoder/encoder_PI/dense_100_PI/bias'autoencoder/encoder_PI/dense_284/kernel%autoencoder/encoder_PI/dense_284/bias'autoencoder/encoder_PI/dense_285/kernel%autoencoder/encoder_PI/dense_285/bias'autoencoder/encoder_PI/dense_286/kernel%autoencoder/encoder_PI/dense_286/bias0autoencoder/encoder_image/dense_100_image/kernel.autoencoder/encoder_image/dense_100_image/bias*autoencoder/encoder_image/dense_291/kernel(autoencoder/encoder_image/dense_291/bias*autoencoder/encoder_image/dense_292/kernel(autoencoder/encoder_image/dense_292/bias'autoencoder/decoder_PC/dense_289/kernel%autoencoder/decoder_PC/dense_289/bias'autoencoder/decoder_PC/dense_290/kernel%autoencoder/decoder_PC/dense_290/bias'autoencoder/decoder_PI/dense_287/kernel%autoencoder/decoder_PI/dense_287/bias'autoencoder/decoder_PI/dense_288/kernel%autoencoder/decoder_PI/dense_288/bias*autoencoder/decoder_image/dense_293/kernel(autoencoder/decoder_image/dense_293/bias*autoencoder/decoder_image/dense_294/kernel(autoencoder/decoder_image/dense_294/bias*autoencoder/decoder_image/dense_295/kernel(autoencoder/decoder_image/dense_295/bias*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *0
f+R)
'__inference_signature_wrapper_146872652
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename>autoencoder/encoder_PI/dense_100_PI/kernel/Read/ReadVariableOp<autoencoder/encoder_PI/dense_100_PI/bias/Read/ReadVariableOp;autoencoder/encoder_PI/dense_284/kernel/Read/ReadVariableOp9autoencoder/encoder_PI/dense_284/bias/Read/ReadVariableOp;autoencoder/encoder_PI/dense_285/kernel/Read/ReadVariableOp9autoencoder/encoder_PI/dense_285/bias/Read/ReadVariableOp;autoencoder/encoder_PI/dense_286/kernel/Read/ReadVariableOp9autoencoder/encoder_PI/dense_286/bias/Read/ReadVariableOp;autoencoder/decoder_PI/dense_287/kernel/Read/ReadVariableOp9autoencoder/decoder_PI/dense_287/bias/Read/ReadVariableOp;autoencoder/decoder_PI/dense_288/kernel/Read/ReadVariableOp9autoencoder/decoder_PI/dense_288/bias/Read/ReadVariableOp;autoencoder/decoder_PC/dense_289/kernel/Read/ReadVariableOp9autoencoder/decoder_PC/dense_289/bias/Read/ReadVariableOp;autoencoder/decoder_PC/dense_290/kernel/Read/ReadVariableOp9autoencoder/decoder_PC/dense_290/bias/Read/ReadVariableOpDautoencoder/encoder_image/dense_100_image/kernel/Read/ReadVariableOpBautoencoder/encoder_image/dense_100_image/bias/Read/ReadVariableOp>autoencoder/encoder_image/dense_291/kernel/Read/ReadVariableOp<autoencoder/encoder_image/dense_291/bias/Read/ReadVariableOp>autoencoder/encoder_image/dense_292/kernel/Read/ReadVariableOp<autoencoder/encoder_image/dense_292/bias/Read/ReadVariableOp>autoencoder/decoder_image/dense_293/kernel/Read/ReadVariableOp<autoencoder/decoder_image/dense_293/bias/Read/ReadVariableOp>autoencoder/decoder_image/dense_294/kernel/Read/ReadVariableOp<autoencoder/decoder_image/dense_294/bias/Read/ReadVariableOp>autoencoder/decoder_image/dense_295/kernel/Read/ReadVariableOp<autoencoder/decoder_image/dense_295/bias/Read/ReadVariableOpConst*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_save_146872988
Î
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*autoencoder/encoder_PI/dense_100_PI/kernel(autoencoder/encoder_PI/dense_100_PI/bias'autoencoder/encoder_PI/dense_284/kernel%autoencoder/encoder_PI/dense_284/bias'autoencoder/encoder_PI/dense_285/kernel%autoencoder/encoder_PI/dense_285/bias'autoencoder/encoder_PI/dense_286/kernel%autoencoder/encoder_PI/dense_286/bias'autoencoder/decoder_PI/dense_287/kernel%autoencoder/decoder_PI/dense_287/bias'autoencoder/decoder_PI/dense_288/kernel%autoencoder/decoder_PI/dense_288/bias'autoencoder/decoder_PC/dense_289/kernel%autoencoder/decoder_PC/dense_289/bias'autoencoder/decoder_PC/dense_290/kernel%autoencoder/decoder_PC/dense_290/bias0autoencoder/encoder_image/dense_100_image/kernel.autoencoder/encoder_image/dense_100_image/bias*autoencoder/encoder_image/dense_291/kernel(autoencoder/encoder_image/dense_291/bias*autoencoder/encoder_image/dense_292/kernel(autoencoder/encoder_image/dense_292/bias*autoencoder/decoder_image/dense_293/kernel(autoencoder/decoder_image/dense_293/bias*autoencoder/decoder_image/dense_294/kernel(autoencoder/decoder_image/dense_294/bias*autoencoder/decoder_image/dense_295/kernel(autoencoder/decoder_image/dense_295/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference__traced_restore_146873082Âç
+
í
I__inference_encoder_PI_layer_call_and_return_conditional_losses_146872706

inputs>
+dense_100_pi_matmul_readvariableop_resource:	Äd:
,dense_100_pi_biasadd_readvariableop_resource:d:
(dense_284_matmul_readvariableop_resource:d27
)dense_284_biasadd_readvariableop_resource:2:
(dense_285_matmul_readvariableop_resource:27
)dense_285_biasadd_readvariableop_resource::
(dense_286_matmul_readvariableop_resource:27
)dense_286_biasadd_readvariableop_resource:
identity

identity_1¢#dense_100_PI/BiasAdd/ReadVariableOp¢"dense_100_PI/MatMul/ReadVariableOp¢ dense_284/BiasAdd/ReadVariableOp¢dense_284/MatMul/ReadVariableOp¢ dense_285/BiasAdd/ReadVariableOp¢dense_285/MatMul/ReadVariableOp¢ dense_286/BiasAdd/ReadVariableOp¢dense_286/MatMul/ReadVariableOpµ
"dense_100_PI/MatMul/ReadVariableOpReadVariableOp+dense_100_pi_matmul_readvariableop_resource*
_output_shapes
:	Äd*
dtype02$
"dense_100_PI/MatMul/ReadVariableOp
dense_100_PI/MatMulMatMulinputs*dense_100_PI/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_PI/MatMul³
#dense_100_PI/BiasAdd/ReadVariableOpReadVariableOp,dense_100_pi_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#dense_100_PI/BiasAdd/ReadVariableOpµ
dense_100_PI/BiasAddBiasAdddense_100_PI/MatMul:product:0+dense_100_PI/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_PI/BiasAdd
dense_100_PI/ReluReludense_100_PI/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_PI/Relu«
dense_284/MatMul/ReadVariableOpReadVariableOp(dense_284_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02!
dense_284/MatMul/ReadVariableOpª
dense_284/MatMulMatMuldense_100_PI/Relu:activations:0'dense_284/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_284/MatMulª
 dense_284/BiasAdd/ReadVariableOpReadVariableOp)dense_284_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 dense_284/BiasAdd/ReadVariableOp©
dense_284/BiasAddBiasAdddense_284/MatMul:product:0(dense_284/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_284/BiasAddv
dense_284/ReluReludense_284/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_284/Relu«
dense_285/MatMul/ReadVariableOpReadVariableOp(dense_285_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_285/MatMul/ReadVariableOp§
dense_285/MatMulMatMuldense_284/Relu:activations:0'dense_285/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_285/MatMulª
 dense_285/BiasAdd/ReadVariableOpReadVariableOp)dense_285_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_285/BiasAdd/ReadVariableOp©
dense_285/BiasAddBiasAdddense_285/MatMul:product:0(dense_285/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_285/BiasAdd«
dense_286/MatMul/ReadVariableOpReadVariableOp(dense_286_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_286/MatMul/ReadVariableOp§
dense_286/MatMulMatMuldense_284/Relu:activations:0'dense_286/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_286/MatMulª
 dense_286/BiasAdd/ReadVariableOpReadVariableOp)dense_286_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_286/BiasAdd/ReadVariableOp©
dense_286/BiasAddBiasAdddense_286/MatMul:product:0(dense_286/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_286/BiasAdd
IdentityIdentitydense_285/BiasAdd:output:0$^dense_100_PI/BiasAdd/ReadVariableOp#^dense_100_PI/MatMul/ReadVariableOp!^dense_284/BiasAdd/ReadVariableOp ^dense_284/MatMul/ReadVariableOp!^dense_285/BiasAdd/ReadVariableOp ^dense_285/MatMul/ReadVariableOp!^dense_286/BiasAdd/ReadVariableOp ^dense_286/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identitydense_286/BiasAdd:output:0$^dense_100_PI/BiasAdd/ReadVariableOp#^dense_100_PI/MatMul/ReadVariableOp!^dense_284/BiasAdd/ReadVariableOp ^dense_284/MatMul/ReadVariableOp!^dense_285/BiasAdd/ReadVariableOp ^dense_285/MatMul/ReadVariableOp!^dense_286/BiasAdd/ReadVariableOp ^dense_286/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÄ: : : : : : : : 2J
#dense_100_PI/BiasAdd/ReadVariableOp#dense_100_PI/BiasAdd/ReadVariableOp2H
"dense_100_PI/MatMul/ReadVariableOp"dense_100_PI/MatMul/ReadVariableOp2D
 dense_284/BiasAdd/ReadVariableOp dense_284/BiasAdd/ReadVariableOp2B
dense_284/MatMul/ReadVariableOpdense_284/MatMul/ReadVariableOp2D
 dense_285/BiasAdd/ReadVariableOp dense_285/BiasAdd/ReadVariableOp2B
dense_285/MatMul/ReadVariableOpdense_285/MatMul/ReadVariableOp2D
 dense_286/BiasAdd/ReadVariableOp dense_286/BiasAdd/ReadVariableOp2B
dense_286/MatMul/ReadVariableOpdense_286/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
Ò

'__inference_signature_wrapper_146872652
input_1
input_2
unknown:	Äd
	unknown_0:d
	unknown_1:d2
	unknown_2:2
	unknown_3:2
	unknown_4:
	unknown_5:2
	unknown_6:
	unknown_7:
d
	unknown_8:d
	unknown_9:d

unknown_10:

unknown_11:d

unknown_12:

unknown_13:2

unknown_14:2

unknown_15:2

unknown_16:

unknown_17:d

unknown_18:d

unknown_19:	dÄ

unknown_20:	Ä

unknown_21:2

unknown_22:2

unknown_23:2d

unknown_24:d

unknown_25:
d

unknown_26:

identity

identity_1¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__wrapped_model_1468721742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
!
_user_specified_name	input_1:RN
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ü
Ü
I__inference_decoder_PC_layer_call_and_return_conditional_losses_146872342

inputs:
(dense_289_matmul_readvariableop_resource:27
)dense_289_biasadd_readvariableop_resource:2:
(dense_290_matmul_readvariableop_resource:27
)dense_290_biasadd_readvariableop_resource:
identity¢ dense_289/BiasAdd/ReadVariableOp¢dense_289/MatMul/ReadVariableOp¢ dense_290/BiasAdd/ReadVariableOp¢dense_290/MatMul/ReadVariableOp«
dense_289/MatMul/ReadVariableOpReadVariableOp(dense_289_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_289/MatMul/ReadVariableOp
dense_289/MatMulMatMulinputs'dense_289/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_289/MatMulª
 dense_289/BiasAdd/ReadVariableOpReadVariableOp)dense_289_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 dense_289/BiasAdd/ReadVariableOp©
dense_289/BiasAddBiasAdddense_289/MatMul:product:0(dense_289/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_289/BiasAddv
dense_289/ReluReludense_289/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_289/Relu«
dense_290/MatMul/ReadVariableOpReadVariableOp(dense_290_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_290/MatMul/ReadVariableOp§
dense_290/MatMulMatMuldense_289/Relu:activations:0'dense_290/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_290/MatMulª
 dense_290/BiasAdd/ReadVariableOpReadVariableOp)dense_290_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_290/BiasAdd/ReadVariableOp©
dense_290/BiasAddBiasAdddense_290/MatMul:product:0(dense_290/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_290/BiasAddø
IdentityIdentitydense_290/BiasAdd:output:0!^dense_289/BiasAdd/ReadVariableOp ^dense_289/MatMul/ReadVariableOp!^dense_290/BiasAdd/ReadVariableOp ^dense_290/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2D
 dense_289/BiasAdd/ReadVariableOp dense_289/BiasAdd/ReadVariableOp2B
dense_289/MatMul/ReadVariableOpdense_289/MatMul/ReadVariableOp2D
 dense_290/BiasAdd/ReadVariableOp dense_290/BiasAdd/ReadVariableOp2B
dense_290/MatMul/ReadVariableOpdense_290/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨P
Ñ
J__inference_autoencoder_layer_call_and_return_conditional_losses_146872423
input_1
input_2'
encoder_pi_146872211:	Äd"
encoder_pi_146872213:d&
encoder_pi_146872215:d2"
encoder_pi_146872217:2&
encoder_pi_146872219:2"
encoder_pi_146872221:&
encoder_pi_146872223:2"
encoder_pi_146872225:+
encoder_image_146872254:
d%
encoder_image_146872256:d)
encoder_image_146872258:d%
encoder_image_146872260:)
encoder_image_146872262:d%
encoder_image_146872264:&
decoder_pc_146872343:2"
decoder_pc_146872345:2&
decoder_pc_146872347:2"
decoder_pc_146872349:&
decoder_pi_146872373:d"
decoder_pi_146872375:d'
decoder_pi_146872377:	dÄ#
decoder_pi_146872379:	Ä)
decoder_image_146872407:2%
decoder_image_146872409:2)
decoder_image_146872411:2d%
decoder_image_146872413:d+
decoder_image_146872415:
d'
decoder_image_146872417:

identity

identity_1

identity_2¢"decoder_PC/StatefulPartitionedCall¢"decoder_PI/StatefulPartitionedCall¢%decoder_image/StatefulPartitionedCall¢"encoder_PI/StatefulPartitionedCall¢%encoder_image/StatefulPartitionedCall¢#sampling_28/StatefulPartitionedCallÏ
"encoder_PI/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_pi_146872211encoder_pi_146872213encoder_pi_146872215encoder_pi_146872217encoder_pi_146872219encoder_pi_146872221encoder_pi_146872223encoder_pi_146872225*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_encoder_PI_layer_call_and_return_conditional_losses_1468722102$
"encoder_PI/StatefulPartitionedCallº
%encoder_image/StatefulPartitionedCallStatefulPartitionedCallinput_2encoder_image_146872254encoder_image_146872256encoder_image_146872258encoder_image_146872260encoder_image_146872262encoder_image_146872264*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_image_layer_call_and_return_conditional_losses_1468722532'
%encoder_image/StatefulPartitionedCall©
truedivRealDiv+encoder_PI/StatefulPartitionedCall:output:0+encoder_PI/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
truediv³
	truediv_1RealDiv.encoder_image/StatefulPartitionedCall:output:0.encoder_image/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_1a
addAddV2truediv:z:0truediv_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_
truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
truediv_2/x
	truediv_2RealDivtruediv_2/x:output:0+encoder_PI/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_2W
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xj
add_1AddV2add_1/x:output:0truediv_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1_
truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
truediv_3/x
	truediv_3RealDivtruediv_3/x:output:0.encoder_image/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_3c
add_2AddV2	add_1:z:0truediv_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_2W
mulMuladd:z:0	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_
truediv_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
truediv_4/x
	truediv_4RealDivtruediv_4/x:output:0+encoder_PI/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_4W
add_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_3/xj
add_3AddV2add_3/x:output:0truediv_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_3_
truediv_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
truediv_5/x
	truediv_5RealDivtruediv_5/x:output:0.encoder_image/StatefulPartitionedCall:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	truediv_5c
add_4AddV2	add_3:z:0truediv_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_4
#sampling_28/StatefulPartitionedCallStatefulPartitionedCallmul:z:0	add_4:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sampling_28_layer_call_and_return_conditional_losses_1468723092%
#sampling_28/StatefulPartitionedCallW
SquareSquare	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
SquareO
LogLog
Square:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
LogY
Square_1Squaremul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Square_1Z
subSubLog:y:0Square_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub[
Square_2Square	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Square_2^
sub_1Subsub:z:0Square_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sub_1W
add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_5/yf
add_5AddV2	sub_1:z:0add_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_5r
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicese
MeanMean	add_5:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
MeanW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2	
mul_1/x[
mul_1Mulmul_1/x:output:0Mean:output:0*
T0*
_output_shapes
:2
mul_1X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstM
SumSum	mul_1:z:0Const:output:0*
T0*
_output_shapes
: 2
Sum
"decoder_PC/StatefulPartitionedCallStatefulPartitionedCall,sampling_28/StatefulPartitionedCall:output:0decoder_pc_146872343decoder_pc_146872345decoder_pc_146872347decoder_pc_146872349*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_PC_layer_call_and_return_conditional_losses_1468723422$
"decoder_PC/StatefulPartitionedCalld
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim´
splitSplitsplit/split_dim:output:0+decoder_PC/StatefulPartitionedCall:output:0*
T0*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
splitã
"decoder_PI/StatefulPartitionedCallStatefulPartitionedCallsplit:output:0decoder_pi_146872373decoder_pi_146872375decoder_pi_146872377decoder_pi_146872379*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_PI_layer_call_and_return_conditional_losses_1468723722$
"decoder_PI/StatefulPartitionedCall¯
%decoder_image/StatefulPartitionedCallStatefulPartitionedCallsplit:output:1decoder_image_146872407decoder_image_146872409decoder_image_146872411decoder_image_146872413decoder_image_146872415decoder_image_146872417*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_decoder_image_layer_call_and_return_conditional_losses_1468724062'
%decoder_image/StatefulPartitionedCallå
IdentityIdentity+decoder_PI/StatefulPartitionedCall:output:0#^decoder_PC/StatefulPartitionedCall#^decoder_PI/StatefulPartitionedCall&^decoder_image/StatefulPartitionedCall#^encoder_PI/StatefulPartitionedCall&^encoder_image/StatefulPartitionedCall$^sampling_28/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identityí

Identity_1Identity.decoder_image/StatefulPartitionedCall:output:0#^decoder_PC/StatefulPartitionedCall#^decoder_PI/StatefulPartitionedCall&^decoder_image/StatefulPartitionedCall#^encoder_PI/StatefulPartitionedCall&^encoder_image/StatefulPartitionedCall$^sampling_28/StatefulPartitionedCall*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1¸

Identity_2IdentitySum:output:0#^decoder_PC/StatefulPartitionedCall#^decoder_PI/StatefulPartitionedCall&^decoder_image/StatefulPartitionedCall#^encoder_PI/StatefulPartitionedCall&^encoder_image/StatefulPartitionedCall$^sampling_28/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"decoder_PC/StatefulPartitionedCall"decoder_PC/StatefulPartitionedCall2H
"decoder_PI/StatefulPartitionedCall"decoder_PI/StatefulPartitionedCall2N
%decoder_image/StatefulPartitionedCall%decoder_image/StatefulPartitionedCall2H
"encoder_PI/StatefulPartitionedCall"encoder_PI/StatefulPartitionedCall2N
%encoder_image/StatefulPartitionedCall%encoder_image/StatefulPartitionedCall2J
#sampling_28/StatefulPartitionedCall#sampling_28/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
!
_user_specified_name	input_1:RN
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ü
Ü
I__inference_decoder_PC_layer_call_and_return_conditional_losses_146872795

inputs:
(dense_289_matmul_readvariableop_resource:27
)dense_289_biasadd_readvariableop_resource:2:
(dense_290_matmul_readvariableop_resource:27
)dense_290_biasadd_readvariableop_resource:
identity¢ dense_289/BiasAdd/ReadVariableOp¢dense_289/MatMul/ReadVariableOp¢ dense_290/BiasAdd/ReadVariableOp¢dense_290/MatMul/ReadVariableOp«
dense_289/MatMul/ReadVariableOpReadVariableOp(dense_289_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_289/MatMul/ReadVariableOp
dense_289/MatMulMatMulinputs'dense_289/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_289/MatMulª
 dense_289/BiasAdd/ReadVariableOpReadVariableOp)dense_289_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 dense_289/BiasAdd/ReadVariableOp©
dense_289/BiasAddBiasAdddense_289/MatMul:product:0(dense_289/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_289/BiasAddv
dense_289/ReluReludense_289/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_289/Relu«
dense_290/MatMul/ReadVariableOpReadVariableOp(dense_290_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_290/MatMul/ReadVariableOp§
dense_290/MatMulMatMuldense_289/Relu:activations:0'dense_290/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_290/MatMulª
 dense_290/BiasAdd/ReadVariableOpReadVariableOp)dense_290_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_290/BiasAdd/ReadVariableOp©
dense_290/BiasAddBiasAdddense_290/MatMul:product:0(dense_290/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_290/BiasAddø
IdentityIdentitydense_290/BiasAdd:output:0!^dense_289/BiasAdd/ReadVariableOp ^dense_289/MatMul/ReadVariableOp!^dense_290/BiasAdd/ReadVariableOp ^dense_290/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2D
 dense_289/BiasAdd/ReadVariableOp dense_289/BiasAdd/ReadVariableOp2B
dense_289/MatMul/ReadVariableOpdense_289/MatMul/ReadVariableOp2D
 dense_290/BiasAdd/ReadVariableOp dense_290/BiasAdd/ReadVariableOp2B
dense_290/MatMul/ReadVariableOpdense_290/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
«
%__inference__traced_restore_146873082
file_prefixN
;assignvariableop_autoencoder_encoder_pi_dense_100_pi_kernel:	ÄdI
;assignvariableop_1_autoencoder_encoder_pi_dense_100_pi_bias:dL
:assignvariableop_2_autoencoder_encoder_pi_dense_284_kernel:d2F
8assignvariableop_3_autoencoder_encoder_pi_dense_284_bias:2L
:assignvariableop_4_autoencoder_encoder_pi_dense_285_kernel:2F
8assignvariableop_5_autoencoder_encoder_pi_dense_285_bias:L
:assignvariableop_6_autoencoder_encoder_pi_dense_286_kernel:2F
8assignvariableop_7_autoencoder_encoder_pi_dense_286_bias:L
:assignvariableop_8_autoencoder_decoder_pi_dense_287_kernel:dF
8assignvariableop_9_autoencoder_decoder_pi_dense_287_bias:dN
;assignvariableop_10_autoencoder_decoder_pi_dense_288_kernel:	dÄH
9assignvariableop_11_autoencoder_decoder_pi_dense_288_bias:	ÄM
;assignvariableop_12_autoencoder_decoder_pc_dense_289_kernel:2G
9assignvariableop_13_autoencoder_decoder_pc_dense_289_bias:2M
;assignvariableop_14_autoencoder_decoder_pc_dense_290_kernel:2G
9assignvariableop_15_autoencoder_decoder_pc_dense_290_bias:X
Dassignvariableop_16_autoencoder_encoder_image_dense_100_image_kernel:
dP
Bassignvariableop_17_autoencoder_encoder_image_dense_100_image_bias:dP
>assignvariableop_18_autoencoder_encoder_image_dense_291_kernel:dJ
<assignvariableop_19_autoencoder_encoder_image_dense_291_bias:P
>assignvariableop_20_autoencoder_encoder_image_dense_292_kernel:dJ
<assignvariableop_21_autoencoder_encoder_image_dense_292_bias:P
>assignvariableop_22_autoencoder_decoder_image_dense_293_kernel:2J
<assignvariableop_23_autoencoder_decoder_image_dense_293_bias:2P
>assignvariableop_24_autoencoder_decoder_image_dense_294_kernel:2dJ
<assignvariableop_25_autoencoder_decoder_image_dense_294_bias:dR
>assignvariableop_26_autoencoder_decoder_image_dense_295_kernel:
dL
<assignvariableop_27_autoencoder_decoder_image_dense_295_bias:

identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9±
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*½
value³B°B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices½
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityº
AssignVariableOpAssignVariableOp;assignvariableop_autoencoder_encoder_pi_dense_100_pi_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1À
AssignVariableOp_1AssignVariableOp;assignvariableop_1_autoencoder_encoder_pi_dense_100_pi_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¿
AssignVariableOp_2AssignVariableOp:assignvariableop_2_autoencoder_encoder_pi_dense_284_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3½
AssignVariableOp_3AssignVariableOp8assignvariableop_3_autoencoder_encoder_pi_dense_284_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¿
AssignVariableOp_4AssignVariableOp:assignvariableop_4_autoencoder_encoder_pi_dense_285_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5½
AssignVariableOp_5AssignVariableOp8assignvariableop_5_autoencoder_encoder_pi_dense_285_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¿
AssignVariableOp_6AssignVariableOp:assignvariableop_6_autoencoder_encoder_pi_dense_286_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7½
AssignVariableOp_7AssignVariableOp8assignvariableop_7_autoencoder_encoder_pi_dense_286_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¿
AssignVariableOp_8AssignVariableOp:assignvariableop_8_autoencoder_decoder_pi_dense_287_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9½
AssignVariableOp_9AssignVariableOp8assignvariableop_9_autoencoder_decoder_pi_dense_287_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ã
AssignVariableOp_10AssignVariableOp;assignvariableop_10_autoencoder_decoder_pi_dense_288_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_autoencoder_decoder_pi_dense_288_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ã
AssignVariableOp_12AssignVariableOp;assignvariableop_12_autoencoder_decoder_pc_dense_289_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Á
AssignVariableOp_13AssignVariableOp9assignvariableop_13_autoencoder_decoder_pc_dense_289_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ã
AssignVariableOp_14AssignVariableOp;assignvariableop_14_autoencoder_decoder_pc_dense_290_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Á
AssignVariableOp_15AssignVariableOp9assignvariableop_15_autoencoder_decoder_pc_dense_290_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ì
AssignVariableOp_16AssignVariableOpDassignvariableop_16_autoencoder_encoder_image_dense_100_image_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ê
AssignVariableOp_17AssignVariableOpBassignvariableop_17_autoencoder_encoder_image_dense_100_image_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Æ
AssignVariableOp_18AssignVariableOp>assignvariableop_18_autoencoder_encoder_image_dense_291_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ä
AssignVariableOp_19AssignVariableOp<assignvariableop_19_autoencoder_encoder_image_dense_291_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Æ
AssignVariableOp_20AssignVariableOp>assignvariableop_20_autoencoder_encoder_image_dense_292_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ä
AssignVariableOp_21AssignVariableOp<assignvariableop_21_autoencoder_encoder_image_dense_292_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Æ
AssignVariableOp_22AssignVariableOp>assignvariableop_22_autoencoder_decoder_image_dense_293_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ä
AssignVariableOp_23AssignVariableOp<assignvariableop_23_autoencoder_decoder_image_dense_293_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Æ
AssignVariableOp_24AssignVariableOp>assignvariableop_24_autoencoder_decoder_image_dense_294_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ä
AssignVariableOp_25AssignVariableOp<assignvariableop_25_autoencoder_decoder_image_dense_294_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Æ
AssignVariableOp_26AssignVariableOp>assignvariableop_26_autoencoder_decoder_image_dense_295_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ä
AssignVariableOp_27AssignVariableOp<assignvariableop_27_autoencoder_decoder_image_dense_295_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÆ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28¹
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¨
w
J__inference_sampling_28_layer_call_and_return_conditional_losses_146872309

inputs
inputs_1
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevä
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ý2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal`
mulMulinputs_1random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulV
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
y
J__inference_sampling_28_layer_call_and_return_conditional_losses_146872765
inputs_0
inputs_1
identityF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevå
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2õô¢2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal`
mulMulinputs_1random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulX
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ñ!
Ã
L__inference_encoder_image_layer_call_and_return_conditional_losses_146872253

inputsB
.dense_100_image_matmul_readvariableop_resource:
d=
/dense_100_image_biasadd_readvariableop_resource:d:
(dense_291_matmul_readvariableop_resource:d7
)dense_291_biasadd_readvariableop_resource::
(dense_292_matmul_readvariableop_resource:d7
)dense_292_biasadd_readvariableop_resource:
identity

identity_1¢&dense_100_image/BiasAdd/ReadVariableOp¢%dense_100_image/MatMul/ReadVariableOp¢ dense_291/BiasAdd/ReadVariableOp¢dense_291/MatMul/ReadVariableOp¢ dense_292/BiasAdd/ReadVariableOp¢dense_292/MatMul/ReadVariableOp¿
%dense_100_image/MatMul/ReadVariableOpReadVariableOp.dense_100_image_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02'
%dense_100_image/MatMul/ReadVariableOp£
dense_100_image/MatMulMatMulinputs-dense_100_image/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_image/MatMul¼
&dense_100_image/BiasAdd/ReadVariableOpReadVariableOp/dense_100_image_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02(
&dense_100_image/BiasAdd/ReadVariableOpÁ
dense_100_image/BiasAddBiasAdd dense_100_image/MatMul:product:0.dense_100_image/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_image/BiasAdd
dense_100_image/ReluRelu dense_100_image/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_image/Relu«
dense_291/MatMul/ReadVariableOpReadVariableOp(dense_291_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_291/MatMul/ReadVariableOp­
dense_291/MatMulMatMul"dense_100_image/Relu:activations:0'dense_291/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_291/MatMulª
 dense_291/BiasAdd/ReadVariableOpReadVariableOp)dense_291_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_291/BiasAdd/ReadVariableOp©
dense_291/BiasAddBiasAdddense_291/MatMul:product:0(dense_291/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_291/BiasAdd«
dense_292/MatMul/ReadVariableOpReadVariableOp(dense_292_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_292/MatMul/ReadVariableOp­
dense_292/MatMulMatMul"dense_100_image/Relu:activations:0'dense_292/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_292/MatMulª
 dense_292/BiasAdd/ReadVariableOpReadVariableOp)dense_292_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_292/BiasAdd/ReadVariableOp©
dense_292/BiasAddBiasAdddense_292/MatMul:product:0(dense_292/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_292/BiasAddÉ
IdentityIdentitydense_291/BiasAdd:output:0'^dense_100_image/BiasAdd/ReadVariableOp&^dense_100_image/MatMul/ReadVariableOp!^dense_291/BiasAdd/ReadVariableOp ^dense_291/MatMul/ReadVariableOp!^dense_292/BiasAdd/ReadVariableOp ^dense_292/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÍ

Identity_1Identitydense_292/BiasAdd:output:0'^dense_100_image/BiasAdd/ReadVariableOp&^dense_100_image/MatMul/ReadVariableOp!^dense_291/BiasAdd/ReadVariableOp ^dense_291/MatMul/ReadVariableOp!^dense_292/BiasAdd/ReadVariableOp ^dense_292/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&dense_100_image/BiasAdd/ReadVariableOp&dense_100_image/BiasAdd/ReadVariableOp2N
%dense_100_image/MatMul/ReadVariableOp%dense_100_image/MatMul/ReadVariableOp2D
 dense_291/BiasAdd/ReadVariableOp dense_291/BiasAdd/ReadVariableOp2B
dense_291/MatMul/ReadVariableOpdense_291/MatMul/ReadVariableOp2D
 dense_292/BiasAdd/ReadVariableOp dense_292/BiasAdd/ReadVariableOp2B
dense_292/MatMul/ReadVariableOpdense_292/MatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È
 
$__inference__wrapped_model_146872174
input_1
input_2U
Bautoencoder_encoder_pi_dense_100_pi_matmul_readvariableop_resource:	ÄdQ
Cautoencoder_encoder_pi_dense_100_pi_biasadd_readvariableop_resource:dQ
?autoencoder_encoder_pi_dense_284_matmul_readvariableop_resource:d2N
@autoencoder_encoder_pi_dense_284_biasadd_readvariableop_resource:2Q
?autoencoder_encoder_pi_dense_285_matmul_readvariableop_resource:2N
@autoencoder_encoder_pi_dense_285_biasadd_readvariableop_resource:Q
?autoencoder_encoder_pi_dense_286_matmul_readvariableop_resource:2N
@autoencoder_encoder_pi_dense_286_biasadd_readvariableop_resource:\
Hautoencoder_encoder_image_dense_100_image_matmul_readvariableop_resource:
dW
Iautoencoder_encoder_image_dense_100_image_biasadd_readvariableop_resource:dT
Bautoencoder_encoder_image_dense_291_matmul_readvariableop_resource:dQ
Cautoencoder_encoder_image_dense_291_biasadd_readvariableop_resource:T
Bautoencoder_encoder_image_dense_292_matmul_readvariableop_resource:dQ
Cautoencoder_encoder_image_dense_292_biasadd_readvariableop_resource:Q
?autoencoder_decoder_pc_dense_289_matmul_readvariableop_resource:2N
@autoencoder_decoder_pc_dense_289_biasadd_readvariableop_resource:2Q
?autoencoder_decoder_pc_dense_290_matmul_readvariableop_resource:2N
@autoencoder_decoder_pc_dense_290_biasadd_readvariableop_resource:Q
?autoencoder_decoder_pi_dense_287_matmul_readvariableop_resource:dN
@autoencoder_decoder_pi_dense_287_biasadd_readvariableop_resource:dR
?autoencoder_decoder_pi_dense_288_matmul_readvariableop_resource:	dÄO
@autoencoder_decoder_pi_dense_288_biasadd_readvariableop_resource:	ÄT
Bautoencoder_decoder_image_dense_293_matmul_readvariableop_resource:2Q
Cautoencoder_decoder_image_dense_293_biasadd_readvariableop_resource:2T
Bautoencoder_decoder_image_dense_294_matmul_readvariableop_resource:2dQ
Cautoencoder_decoder_image_dense_294_biasadd_readvariableop_resource:dV
Bautoencoder_decoder_image_dense_295_matmul_readvariableop_resource:
dS
Cautoencoder_decoder_image_dense_295_biasadd_readvariableop_resource:

identity

identity_1¢7autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp¢6autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOp¢7autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp¢6autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp¢7autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp¢6autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOp¢7autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp¢6autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp¢:autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp¢9autoencoder/decoder_image/dense_293/MatMul/ReadVariableOp¢:autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp¢9autoencoder/decoder_image/dense_294/MatMul/ReadVariableOp¢:autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp¢9autoencoder/decoder_image/dense_295/MatMul/ReadVariableOp¢:autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp¢9autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOp¢7autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp¢6autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp¢7autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp¢6autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp¢7autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp¢6autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOp¢@autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp¢?autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOp¢:autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp¢9autoencoder/encoder_image/dense_291/MatMul/ReadVariableOp¢:autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp¢9autoencoder/encoder_image/dense_292/MatMul/ReadVariableOpú
9autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOpReadVariableOpBautoencoder_encoder_pi_dense_100_pi_matmul_readvariableop_resource*
_output_shapes
:	Äd*
dtype02;
9autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOpà
*autoencoder/encoder_PI/dense_100_PI/MatMulMatMulinput_1Aautoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2,
*autoencoder/encoder_PI/dense_100_PI/MatMulø
:autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOpReadVariableOpCautoencoder_encoder_pi_dense_100_pi_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02<
:autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp
+autoencoder/encoder_PI/dense_100_PI/BiasAddBiasAdd4autoencoder/encoder_PI/dense_100_PI/MatMul:product:0Bautoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2-
+autoencoder/encoder_PI/dense_100_PI/BiasAddÄ
(autoencoder/encoder_PI/dense_100_PI/ReluRelu4autoencoder/encoder_PI/dense_100_PI/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(autoencoder/encoder_PI/dense_100_PI/Reluð
6autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOpReadVariableOp?autoencoder_encoder_pi_dense_284_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype028
6autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp
'autoencoder/encoder_PI/dense_284/MatMulMatMul6autoencoder/encoder_PI/dense_100_PI/Relu:activations:0>autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22)
'autoencoder/encoder_PI/dense_284/MatMulï
7autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_encoder_pi_dense_284_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype029
7autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp
(autoencoder/encoder_PI/dense_284/BiasAddBiasAdd1autoencoder/encoder_PI/dense_284/MatMul:product:0?autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
(autoencoder/encoder_PI/dense_284/BiasAdd»
%autoencoder/encoder_PI/dense_284/ReluRelu1autoencoder/encoder_PI/dense_284/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22'
%autoencoder/encoder_PI/dense_284/Reluð
6autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOpReadVariableOp?autoencoder_encoder_pi_dense_285_matmul_readvariableop_resource*
_output_shapes

:2*
dtype028
6autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp
'autoencoder/encoder_PI/dense_285/MatMulMatMul3autoencoder/encoder_PI/dense_284/Relu:activations:0>autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'autoencoder/encoder_PI/dense_285/MatMulï
7autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_encoder_pi_dense_285_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp
(autoencoder/encoder_PI/dense_285/BiasAddBiasAdd1autoencoder/encoder_PI/dense_285/MatMul:product:0?autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(autoencoder/encoder_PI/dense_285/BiasAddð
6autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOpReadVariableOp?autoencoder_encoder_pi_dense_286_matmul_readvariableop_resource*
_output_shapes

:2*
dtype028
6autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOp
'autoencoder/encoder_PI/dense_286/MatMulMatMul3autoencoder/encoder_PI/dense_284/Relu:activations:0>autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'autoencoder/encoder_PI/dense_286/MatMulï
7autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_encoder_pi_dense_286_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp
(autoencoder/encoder_PI/dense_286/BiasAddBiasAdd1autoencoder/encoder_PI/dense_286/MatMul:product:0?autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(autoencoder/encoder_PI/dense_286/BiasAdd
?autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOpReadVariableOpHautoencoder_encoder_image_dense_100_image_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02A
?autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOpò
0autoencoder/encoder_image/dense_100_image/MatMulMatMulinput_2Gautoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd22
0autoencoder/encoder_image/dense_100_image/MatMul
@autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOpReadVariableOpIautoencoder_encoder_image_dense_100_image_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02B
@autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp©
1autoencoder/encoder_image/dense_100_image/BiasAddBiasAdd:autoencoder/encoder_image/dense_100_image/MatMul:product:0Hautoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd23
1autoencoder/encoder_image/dense_100_image/BiasAddÖ
.autoencoder/encoder_image/dense_100_image/ReluRelu:autoencoder/encoder_image/dense_100_image/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd20
.autoencoder/encoder_image/dense_100_image/Reluù
9autoencoder/encoder_image/dense_291/MatMul/ReadVariableOpReadVariableOpBautoencoder_encoder_image_dense_291_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02;
9autoencoder/encoder_image/dense_291/MatMul/ReadVariableOp
*autoencoder/encoder_image/dense_291/MatMulMatMul<autoencoder/encoder_image/dense_100_image/Relu:activations:0Aautoencoder/encoder_image/dense_291/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*autoencoder/encoder_image/dense_291/MatMulø
:autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOpReadVariableOpCautoencoder_encoder_image_dense_291_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp
+autoencoder/encoder_image/dense_291/BiasAddBiasAdd4autoencoder/encoder_image/dense_291/MatMul:product:0Bautoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+autoencoder/encoder_image/dense_291/BiasAddù
9autoencoder/encoder_image/dense_292/MatMul/ReadVariableOpReadVariableOpBautoencoder_encoder_image_dense_292_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02;
9autoencoder/encoder_image/dense_292/MatMul/ReadVariableOp
*autoencoder/encoder_image/dense_292/MatMulMatMul<autoencoder/encoder_image/dense_100_image/Relu:activations:0Aautoencoder/encoder_image/dense_292/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*autoencoder/encoder_image/dense_292/MatMulø
:autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOpReadVariableOpCautoencoder_encoder_image_dense_292_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp
+autoencoder/encoder_image/dense_292/BiasAddBiasAdd4autoencoder/encoder_image/dense_292/MatMul:product:0Bautoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+autoencoder/encoder_image/dense_292/BiasAddÍ
autoencoder/truedivRealDiv1autoencoder/encoder_PI/dense_285/BiasAdd:output:01autoencoder/encoder_PI/dense_286/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/truediv×
autoencoder/truediv_1RealDiv4autoencoder/encoder_image/dense_291/BiasAdd:output:04autoencoder/encoder_image/dense_292/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/truediv_1
autoencoder/addAddV2autoencoder/truediv:z:0autoencoder/truediv_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/addw
autoencoder/truediv_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/truediv_2/xÀ
autoencoder/truediv_2RealDiv autoencoder/truediv_2/x:output:01autoencoder/encoder_PI/dense_286/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/truediv_2o
autoencoder/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/add_1/x
autoencoder/add_1AddV2autoencoder/add_1/x:output:0autoencoder/truediv_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/add_1w
autoencoder/truediv_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/truediv_3/xÃ
autoencoder/truediv_3RealDiv autoencoder/truediv_3/x:output:04autoencoder/encoder_image/dense_292/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/truediv_3
autoencoder/add_2AddV2autoencoder/add_1:z:0autoencoder/truediv_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/add_2
autoencoder/mulMulautoencoder/add:z:0autoencoder/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/mulw
autoencoder/truediv_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/truediv_4/xÀ
autoencoder/truediv_4RealDiv autoencoder/truediv_4/x:output:01autoencoder/encoder_PI/dense_286/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/truediv_4o
autoencoder/add_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/add_3/x
autoencoder/add_3AddV2autoencoder/add_3/x:output:0autoencoder/truediv_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/add_3w
autoencoder/truediv_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/truediv_5/xÃ
autoencoder/truediv_5RealDiv autoencoder/truediv_5/x:output:04autoencoder/encoder_image/dense_292/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/truediv_5
autoencoder/add_4AddV2autoencoder/add_3:z:0autoencoder/truediv_5:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/add_4
autoencoder/sampling_28/ShapeShapeautoencoder/mul:z:0*
T0*
_output_shapes
:2
autoencoder/sampling_28/Shape¤
+autoencoder/sampling_28/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+autoencoder/sampling_28/strided_slice/stack¨
-autoencoder/sampling_28/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-autoencoder/sampling_28/strided_slice/stack_1¨
-autoencoder/sampling_28/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-autoencoder/sampling_28/strided_slice/stack_2ò
%autoencoder/sampling_28/strided_sliceStridedSlice&autoencoder/sampling_28/Shape:output:04autoencoder/sampling_28/strided_slice/stack:output:06autoencoder/sampling_28/strided_slice/stack_1:output:06autoencoder/sampling_28/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%autoencoder/sampling_28/strided_slice
autoencoder/sampling_28/Shape_1Shapeautoencoder/mul:z:0*
T0*
_output_shapes
:2!
autoencoder/sampling_28/Shape_1¨
-autoencoder/sampling_28/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-autoencoder/sampling_28/strided_slice_1/stack¬
/autoencoder/sampling_28/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/autoencoder/sampling_28/strided_slice_1/stack_1¬
/autoencoder/sampling_28/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/autoencoder/sampling_28/strided_slice_1/stack_2þ
'autoencoder/sampling_28/strided_slice_1StridedSlice(autoencoder/sampling_28/Shape_1:output:06autoencoder/sampling_28/strided_slice_1/stack:output:08autoencoder/sampling_28/strided_slice_1/stack_1:output:08autoencoder/sampling_28/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'autoencoder/sampling_28/strided_slice_1ò
+autoencoder/sampling_28/random_normal/shapePack.autoencoder/sampling_28/strided_slice:output:00autoencoder/sampling_28/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+autoencoder/sampling_28/random_normal/shape
*autoencoder/sampling_28/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*autoencoder/sampling_28/random_normal/mean¡
,autoencoder/sampling_28/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,autoencoder/sampling_28/random_normal/stddev­
:autoencoder/sampling_28/random_normal/RandomStandardNormalRandomStandardNormal4autoencoder/sampling_28/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Þ£2<
:autoencoder/sampling_28/random_normal/RandomStandardNormal
)autoencoder/sampling_28/random_normal/mulMulCautoencoder/sampling_28/random_normal/RandomStandardNormal:output:05autoencoder/sampling_28/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2+
)autoencoder/sampling_28/random_normal/mulô
%autoencoder/sampling_28/random_normalAdd-autoencoder/sampling_28/random_normal/mul:z:03autoencoder/sampling_28/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2'
%autoencoder/sampling_28/random_normalµ
autoencoder/sampling_28/mulMulautoencoder/add_4:z:0)autoencoder/sampling_28/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/sampling_28/mul«
autoencoder/sampling_28/addAddV2autoencoder/mul:z:0autoencoder/sampling_28/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/sampling_28/add{
autoencoder/SquareSquareautoencoder/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/Squares
autoencoder/LogLogautoencoder/Square:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/Log}
autoencoder/Square_1Squareautoencoder/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/Square_1
autoencoder/subSubautoencoder/Log:y:0autoencoder/Square_1:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/sub
autoencoder/Square_2Squareautoencoder/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/Square_2
autoencoder/sub_1Subautoencoder/sub:z:0autoencoder/Square_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/sub_1o
autoencoder/add_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
autoencoder/add_5/y
autoencoder/add_5AddV2autoencoder/sub_1:z:0autoencoder/add_5/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
autoencoder/add_5
"autoencoder/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2$
"autoencoder/Mean/reduction_indices
autoencoder/MeanMeanautoencoder/add_5:z:0+autoencoder/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
autoencoder/Meano
autoencoder/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ¿2
autoencoder/mul_1/x
autoencoder/mul_1Mulautoencoder/mul_1/x:output:0autoencoder/Mean:output:0*
T0*
_output_shapes
:2
autoencoder/mul_1p
autoencoder/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
autoencoder/Const}
autoencoder/SumSumautoencoder/mul_1:z:0autoencoder/Const:output:0*
T0*
_output_shapes
: 2
autoencoder/Sumð
6autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOpReadVariableOp?autoencoder_decoder_pc_dense_289_matmul_readvariableop_resource*
_output_shapes

:2*
dtype028
6autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOpï
'autoencoder/decoder_PC/dense_289/MatMulMatMulautoencoder/sampling_28/add:z:0>autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22)
'autoencoder/decoder_PC/dense_289/MatMulï
7autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_decoder_pc_dense_289_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype029
7autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp
(autoencoder/decoder_PC/dense_289/BiasAddBiasAdd1autoencoder/decoder_PC/dense_289/MatMul:product:0?autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
(autoencoder/decoder_PC/dense_289/BiasAdd»
%autoencoder/decoder_PC/dense_289/ReluRelu1autoencoder/decoder_PC/dense_289/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22'
%autoencoder/decoder_PC/dense_289/Reluð
6autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOpReadVariableOp?autoencoder_decoder_pc_dense_290_matmul_readvariableop_resource*
_output_shapes

:2*
dtype028
6autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp
'autoencoder/decoder_PC/dense_290/MatMulMatMul3autoencoder/decoder_PC/dense_289/Relu:activations:0>autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'autoencoder/decoder_PC/dense_290/MatMulï
7autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_decoder_pc_dense_290_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp
(autoencoder/decoder_PC/dense_290/BiasAddBiasAdd1autoencoder/decoder_PC/dense_290/MatMul:product:0?autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(autoencoder/decoder_PC/dense_290/BiasAdd|
autoencoder/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
autoencoder/split/split_dimÞ
autoencoder/splitSplit$autoencoder/split/split_dim:output:01autoencoder/decoder_PC/dense_290/BiasAdd:output:0*
T0*:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
autoencoder/splitð
6autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOpReadVariableOp?autoencoder_decoder_pi_dense_287_matmul_readvariableop_resource*
_output_shapes

:d*
dtype028
6autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOpê
'autoencoder/decoder_PI/dense_287/MatMulMatMulautoencoder/split:output:0>autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2)
'autoencoder/decoder_PI/dense_287/MatMulï
7autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_decoder_pi_dense_287_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype029
7autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp
(autoencoder/decoder_PI/dense_287/BiasAddBiasAdd1autoencoder/decoder_PI/dense_287/MatMul:product:0?autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(autoencoder/decoder_PI/dense_287/BiasAdd»
%autoencoder/decoder_PI/dense_287/ReluRelu1autoencoder/decoder_PI/dense_287/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2'
%autoencoder/decoder_PI/dense_287/Reluñ
6autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOpReadVariableOp?autoencoder_decoder_pi_dense_288_matmul_readvariableop_resource*
_output_shapes
:	dÄ*
dtype028
6autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp
'autoencoder/decoder_PI/dense_288/MatMulMatMul3autoencoder/decoder_PI/dense_287/Relu:activations:0>autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2)
'autoencoder/decoder_PI/dense_288/MatMulð
7autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOpReadVariableOp@autoencoder_decoder_pi_dense_288_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype029
7autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp
(autoencoder/decoder_PI/dense_288/BiasAddBiasAdd1autoencoder/decoder_PI/dense_288/MatMul:product:0?autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2*
(autoencoder/decoder_PI/dense_288/BiasAddù
9autoencoder/decoder_image/dense_293/MatMul/ReadVariableOpReadVariableOpBautoencoder_decoder_image_dense_293_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02;
9autoencoder/decoder_image/dense_293/MatMul/ReadVariableOpó
*autoencoder/decoder_image/dense_293/MatMulMatMulautoencoder/split:output:1Aautoencoder/decoder_image/dense_293/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22,
*autoencoder/decoder_image/dense_293/MatMulø
:autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOpReadVariableOpCautoencoder_decoder_image_dense_293_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02<
:autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp
+autoencoder/decoder_image/dense_293/BiasAddBiasAdd4autoencoder/decoder_image/dense_293/MatMul:product:0Bautoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22-
+autoencoder/decoder_image/dense_293/BiasAddÄ
(autoencoder/decoder_image/dense_293/ReluRelu4autoencoder/decoder_image/dense_293/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
(autoencoder/decoder_image/dense_293/Reluù
9autoencoder/decoder_image/dense_294/MatMul/ReadVariableOpReadVariableOpBautoencoder_decoder_image_dense_294_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02;
9autoencoder/decoder_image/dense_294/MatMul/ReadVariableOp
*autoencoder/decoder_image/dense_294/MatMulMatMul6autoencoder/decoder_image/dense_293/Relu:activations:0Aautoencoder/decoder_image/dense_294/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2,
*autoencoder/decoder_image/dense_294/MatMulø
:autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOpReadVariableOpCautoencoder_decoder_image_dense_294_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02<
:autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp
+autoencoder/decoder_image/dense_294/BiasAddBiasAdd4autoencoder/decoder_image/dense_294/MatMul:product:0Bautoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2-
+autoencoder/decoder_image/dense_294/BiasAddÄ
(autoencoder/decoder_image/dense_294/ReluRelu4autoencoder/decoder_image/dense_294/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2*
(autoencoder/decoder_image/dense_294/Reluû
9autoencoder/decoder_image/dense_295/MatMul/ReadVariableOpReadVariableOpBautoencoder_decoder_image_dense_295_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02;
9autoencoder/decoder_image/dense_295/MatMul/ReadVariableOp
*autoencoder/decoder_image/dense_295/MatMulMatMul6autoencoder/decoder_image/dense_294/Relu:activations:0Aautoencoder/decoder_image/dense_295/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*autoencoder/decoder_image/dense_295/MatMulú
:autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOpReadVariableOpCautoencoder_decoder_image_dense_295_biasadd_readvariableop_resource*
_output_shapes

:*
dtype02<
:autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp
+autoencoder/decoder_image/dense_295/BiasAddBiasAdd4autoencoder/decoder_image/dense_295/MatMul:product:0Bautoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+autoencoder/decoder_image/dense_295/BiasAdd
IdentityIdentity1autoencoder/decoder_PI/dense_288/BiasAdd:output:08^autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp7^autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOp8^autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp7^autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp8^autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp7^autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOp8^autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp7^autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp;^autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp:^autoencoder/decoder_image/dense_293/MatMul/ReadVariableOp;^autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp:^autoencoder/decoder_image/dense_294/MatMul/ReadVariableOp;^autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp:^autoencoder/decoder_image/dense_295/MatMul/ReadVariableOp;^autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp:^autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOp8^autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp7^autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp8^autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp7^autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp8^autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp7^autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOpA^autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp@^autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOp;^autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp:^autoencoder/encoder_image/dense_291/MatMul/ReadVariableOp;^autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp:^autoencoder/encoder_image/dense_292/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identity

Identity_1Identity4autoencoder/decoder_image/dense_295/BiasAdd:output:08^autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp7^autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOp8^autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp7^autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp8^autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp7^autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOp8^autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp7^autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp;^autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp:^autoencoder/decoder_image/dense_293/MatMul/ReadVariableOp;^autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp:^autoencoder/decoder_image/dense_294/MatMul/ReadVariableOp;^autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp:^autoencoder/decoder_image/dense_295/MatMul/ReadVariableOp;^autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp:^autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOp8^autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp7^autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp8^autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp7^autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp8^autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp7^autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOpA^autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp@^autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOp;^autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp:^autoencoder/encoder_image/dense_291/MatMul/ReadVariableOp;^autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp:^autoencoder/encoder_image/dense_292/MatMul/ReadVariableOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2r
7autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp7autoencoder/decoder_PC/dense_289/BiasAdd/ReadVariableOp2p
6autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOp6autoencoder/decoder_PC/dense_289/MatMul/ReadVariableOp2r
7autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp7autoencoder/decoder_PC/dense_290/BiasAdd/ReadVariableOp2p
6autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp6autoencoder/decoder_PC/dense_290/MatMul/ReadVariableOp2r
7autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp7autoencoder/decoder_PI/dense_287/BiasAdd/ReadVariableOp2p
6autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOp6autoencoder/decoder_PI/dense_287/MatMul/ReadVariableOp2r
7autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp7autoencoder/decoder_PI/dense_288/BiasAdd/ReadVariableOp2p
6autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp6autoencoder/decoder_PI/dense_288/MatMul/ReadVariableOp2x
:autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp:autoencoder/decoder_image/dense_293/BiasAdd/ReadVariableOp2v
9autoencoder/decoder_image/dense_293/MatMul/ReadVariableOp9autoencoder/decoder_image/dense_293/MatMul/ReadVariableOp2x
:autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp:autoencoder/decoder_image/dense_294/BiasAdd/ReadVariableOp2v
9autoencoder/decoder_image/dense_294/MatMul/ReadVariableOp9autoencoder/decoder_image/dense_294/MatMul/ReadVariableOp2x
:autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp:autoencoder/decoder_image/dense_295/BiasAdd/ReadVariableOp2v
9autoencoder/decoder_image/dense_295/MatMul/ReadVariableOp9autoencoder/decoder_image/dense_295/MatMul/ReadVariableOp2x
:autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp:autoencoder/encoder_PI/dense_100_PI/BiasAdd/ReadVariableOp2v
9autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOp9autoencoder/encoder_PI/dense_100_PI/MatMul/ReadVariableOp2r
7autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp7autoencoder/encoder_PI/dense_284/BiasAdd/ReadVariableOp2p
6autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp6autoencoder/encoder_PI/dense_284/MatMul/ReadVariableOp2r
7autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp7autoencoder/encoder_PI/dense_285/BiasAdd/ReadVariableOp2p
6autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp6autoencoder/encoder_PI/dense_285/MatMul/ReadVariableOp2r
7autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp7autoencoder/encoder_PI/dense_286/BiasAdd/ReadVariableOp2p
6autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOp6autoencoder/encoder_PI/dense_286/MatMul/ReadVariableOp2
@autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp@autoencoder/encoder_image/dense_100_image/BiasAdd/ReadVariableOp2
?autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOp?autoencoder/encoder_image/dense_100_image/MatMul/ReadVariableOp2x
:autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp:autoencoder/encoder_image/dense_291/BiasAdd/ReadVariableOp2v
9autoencoder/encoder_image/dense_291/MatMul/ReadVariableOp9autoencoder/encoder_image/dense_291/MatMul/ReadVariableOp2x
:autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp:autoencoder/encoder_image/dense_292/BiasAdd/ReadVariableOp2v
9autoencoder/encoder_image/dense_292/MatMul/ReadVariableOp9autoencoder/encoder_image/dense_292/MatMul/ReadVariableOp:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
!
_user_specified_name	input_1:RN
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

Î
.__inference_encoder_PI_layer_call_fn_146872675

inputs
unknown:	Äd
	unknown_0:d
	unknown_1:d2
	unknown_2:2
	unknown_3:2
	unknown_4:
	unknown_5:2
	unknown_6:
identity

identity_1¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_encoder_PI_layer_call_and_return_conditional_losses_1468722102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÄ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs

Þ
I__inference_decoder_PI_layer_call_and_return_conditional_losses_146872736

inputs:
(dense_287_matmul_readvariableop_resource:d7
)dense_287_biasadd_readvariableop_resource:d;
(dense_288_matmul_readvariableop_resource:	dÄ8
)dense_288_biasadd_readvariableop_resource:	Ä
identity¢ dense_287/BiasAdd/ReadVariableOp¢dense_287/MatMul/ReadVariableOp¢ dense_288/BiasAdd/ReadVariableOp¢dense_288/MatMul/ReadVariableOp«
dense_287/MatMul/ReadVariableOpReadVariableOp(dense_287_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_287/MatMul/ReadVariableOp
dense_287/MatMulMatMulinputs'dense_287/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_287/MatMulª
 dense_287/BiasAdd/ReadVariableOpReadVariableOp)dense_287_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_287/BiasAdd/ReadVariableOp©
dense_287/BiasAddBiasAdddense_287/MatMul:product:0(dense_287/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_287/BiasAddv
dense_287/ReluReludense_287/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_287/Relu¬
dense_288/MatMul/ReadVariableOpReadVariableOp(dense_288_matmul_readvariableop_resource*
_output_shapes
:	dÄ*
dtype02!
dense_288/MatMul/ReadVariableOp¨
dense_288/MatMulMatMuldense_287/Relu:activations:0'dense_288/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2
dense_288/MatMul«
 dense_288/BiasAdd/ReadVariableOpReadVariableOp)dense_288_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype02"
 dense_288/BiasAdd/ReadVariableOpª
dense_288/BiasAddBiasAdddense_288/MatMul:product:0(dense_288/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2
dense_288/BiasAddù
IdentityIdentitydense_288/BiasAdd:output:0!^dense_287/BiasAdd/ReadVariableOp ^dense_287/MatMul/ReadVariableOp!^dense_288/BiasAdd/ReadVariableOp ^dense_288/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2D
 dense_287/BiasAdd/ReadVariableOp dense_287/BiasAdd/ReadVariableOp2B
dense_287/MatMul/ReadVariableOpdense_287/MatMul/ReadVariableOp2D
 dense_288/BiasAdd/ReadVariableOp dense_288/BiasAdd/ReadVariableOp2B
dense_288/MatMul/ReadVariableOpdense_288/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ

1__inference_decoder_image_layer_call_fn_146872855

inputs
unknown:2
	unknown_0:2
	unknown_1:2d
	unknown_2:d
	unknown_3:
d
	unknown_4:

identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_decoder_image_layer_call_and_return_conditional_losses_1468724062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸


1__inference_encoder_image_layer_call_fn_146872814

inputs
unknown:
d
	unknown_0:d
	unknown_1:d
	unknown_2:
	unknown_3:d
	unknown_4:
identity

identity_1¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_encoder_image_layer_call_and_return_conditional_losses_1468722532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Þ
I__inference_decoder_PI_layer_call_and_return_conditional_losses_146872372

inputs:
(dense_287_matmul_readvariableop_resource:d7
)dense_287_biasadd_readvariableop_resource:d;
(dense_288_matmul_readvariableop_resource:	dÄ8
)dense_288_biasadd_readvariableop_resource:	Ä
identity¢ dense_287/BiasAdd/ReadVariableOp¢dense_287/MatMul/ReadVariableOp¢ dense_288/BiasAdd/ReadVariableOp¢dense_288/MatMul/ReadVariableOp«
dense_287/MatMul/ReadVariableOpReadVariableOp(dense_287_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_287/MatMul/ReadVariableOp
dense_287/MatMulMatMulinputs'dense_287/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_287/MatMulª
 dense_287/BiasAdd/ReadVariableOpReadVariableOp)dense_287_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_287/BiasAdd/ReadVariableOp©
dense_287/BiasAddBiasAdddense_287/MatMul:product:0(dense_287/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_287/BiasAddv
dense_287/ReluReludense_287/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_287/Relu¬
dense_288/MatMul/ReadVariableOpReadVariableOp(dense_288_matmul_readvariableop_resource*
_output_shapes
:	dÄ*
dtype02!
dense_288/MatMul/ReadVariableOp¨
dense_288/MatMulMatMuldense_287/Relu:activations:0'dense_288/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2
dense_288/MatMul«
 dense_288/BiasAdd/ReadVariableOpReadVariableOp)dense_288_biasadd_readvariableop_resource*
_output_shapes	
:Ä*
dtype02"
 dense_288/BiasAdd/ReadVariableOpª
dense_288/BiasAddBiasAdddense_288/MatMul:product:0(dense_288/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2
dense_288/BiasAddù
IdentityIdentitydense_288/BiasAdd:output:0!^dense_287/BiasAdd/ReadVariableOp ^dense_287/MatMul/ReadVariableOp!^dense_288/BiasAdd/ReadVariableOp ^dense_288/MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2D
 dense_287/BiasAdd/ReadVariableOp dense_287/BiasAdd/ReadVariableOp2B
dense_287/MatMul/ReadVariableOpdense_287/MatMul/ReadVariableOp2D
 dense_288/BiasAdd/ReadVariableOp dense_288/BiasAdd/ReadVariableOp2B
dense_288/MatMul/ReadVariableOpdense_288/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
+
í
I__inference_encoder_PI_layer_call_and_return_conditional_losses_146872210

inputs>
+dense_100_pi_matmul_readvariableop_resource:	Äd:
,dense_100_pi_biasadd_readvariableop_resource:d:
(dense_284_matmul_readvariableop_resource:d27
)dense_284_biasadd_readvariableop_resource:2:
(dense_285_matmul_readvariableop_resource:27
)dense_285_biasadd_readvariableop_resource::
(dense_286_matmul_readvariableop_resource:27
)dense_286_biasadd_readvariableop_resource:
identity

identity_1¢#dense_100_PI/BiasAdd/ReadVariableOp¢"dense_100_PI/MatMul/ReadVariableOp¢ dense_284/BiasAdd/ReadVariableOp¢dense_284/MatMul/ReadVariableOp¢ dense_285/BiasAdd/ReadVariableOp¢dense_285/MatMul/ReadVariableOp¢ dense_286/BiasAdd/ReadVariableOp¢dense_286/MatMul/ReadVariableOpµ
"dense_100_PI/MatMul/ReadVariableOpReadVariableOp+dense_100_pi_matmul_readvariableop_resource*
_output_shapes
:	Äd*
dtype02$
"dense_100_PI/MatMul/ReadVariableOp
dense_100_PI/MatMulMatMulinputs*dense_100_PI/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_PI/MatMul³
#dense_100_PI/BiasAdd/ReadVariableOpReadVariableOp,dense_100_pi_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02%
#dense_100_PI/BiasAdd/ReadVariableOpµ
dense_100_PI/BiasAddBiasAdddense_100_PI/MatMul:product:0+dense_100_PI/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_PI/BiasAdd
dense_100_PI/ReluReludense_100_PI/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_PI/Relu«
dense_284/MatMul/ReadVariableOpReadVariableOp(dense_284_matmul_readvariableop_resource*
_output_shapes

:d2*
dtype02!
dense_284/MatMul/ReadVariableOpª
dense_284/MatMulMatMuldense_100_PI/Relu:activations:0'dense_284/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_284/MatMulª
 dense_284/BiasAdd/ReadVariableOpReadVariableOp)dense_284_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 dense_284/BiasAdd/ReadVariableOp©
dense_284/BiasAddBiasAdddense_284/MatMul:product:0(dense_284/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_284/BiasAddv
dense_284/ReluReludense_284/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_284/Relu«
dense_285/MatMul/ReadVariableOpReadVariableOp(dense_285_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_285/MatMul/ReadVariableOp§
dense_285/MatMulMatMuldense_284/Relu:activations:0'dense_285/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_285/MatMulª
 dense_285/BiasAdd/ReadVariableOpReadVariableOp)dense_285_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_285/BiasAdd/ReadVariableOp©
dense_285/BiasAddBiasAdddense_285/MatMul:product:0(dense_285/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_285/BiasAdd«
dense_286/MatMul/ReadVariableOpReadVariableOp(dense_286_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_286/MatMul/ReadVariableOp§
dense_286/MatMulMatMuldense_284/Relu:activations:0'dense_286/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_286/MatMulª
 dense_286/BiasAdd/ReadVariableOpReadVariableOp)dense_286_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_286/BiasAdd/ReadVariableOp©
dense_286/BiasAddBiasAdddense_286/MatMul:product:0(dense_286/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_286/BiasAdd
IdentityIdentitydense_285/BiasAdd:output:0$^dense_100_PI/BiasAdd/ReadVariableOp#^dense_100_PI/MatMul/ReadVariableOp!^dense_284/BiasAdd/ReadVariableOp ^dense_284/MatMul/ReadVariableOp!^dense_285/BiasAdd/ReadVariableOp ^dense_285/MatMul/ReadVariableOp!^dense_286/BiasAdd/ReadVariableOp ^dense_286/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identitydense_286/BiasAdd:output:0$^dense_100_PI/BiasAdd/ReadVariableOp#^dense_100_PI/MatMul/ReadVariableOp!^dense_284/BiasAdd/ReadVariableOp ^dense_284/MatMul/ReadVariableOp!^dense_285/BiasAdd/ReadVariableOp ^dense_285/MatMul/ReadVariableOp!^dense_286/BiasAdd/ReadVariableOp ^dense_286/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿÄ: : : : : : : : 2J
#dense_100_PI/BiasAdd/ReadVariableOp#dense_100_PI/BiasAdd/ReadVariableOp2H
"dense_100_PI/MatMul/ReadVariableOp"dense_100_PI/MatMul/ReadVariableOp2D
 dense_284/BiasAdd/ReadVariableOp dense_284/BiasAdd/ReadVariableOp2B
dense_284/MatMul/ReadVariableOpdense_284/MatMul/ReadVariableOp2D
 dense_285/BiasAdd/ReadVariableOp dense_285/BiasAdd/ReadVariableOp2B
dense_285/MatMul/ReadVariableOpdense_285/MatMul/ReadVariableOp2D
 dense_286/BiasAdd/ReadVariableOp dense_286/BiasAdd/ReadVariableOp2B
dense_286/MatMul/ReadVariableOpdense_286/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
 
_user_specified_nameinputs
Ã

L__inference_decoder_image_layer_call_and_return_conditional_losses_146872879

inputs:
(dense_293_matmul_readvariableop_resource:27
)dense_293_biasadd_readvariableop_resource:2:
(dense_294_matmul_readvariableop_resource:2d7
)dense_294_biasadd_readvariableop_resource:d<
(dense_295_matmul_readvariableop_resource:
d9
)dense_295_biasadd_readvariableop_resource:

identity¢ dense_293/BiasAdd/ReadVariableOp¢dense_293/MatMul/ReadVariableOp¢ dense_294/BiasAdd/ReadVariableOp¢dense_294/MatMul/ReadVariableOp¢ dense_295/BiasAdd/ReadVariableOp¢dense_295/MatMul/ReadVariableOp«
dense_293/MatMul/ReadVariableOpReadVariableOp(dense_293_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_293/MatMul/ReadVariableOp
dense_293/MatMulMatMulinputs'dense_293/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_293/MatMulª
 dense_293/BiasAdd/ReadVariableOpReadVariableOp)dense_293_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 dense_293/BiasAdd/ReadVariableOp©
dense_293/BiasAddBiasAdddense_293/MatMul:product:0(dense_293/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_293/BiasAddv
dense_293/ReluReludense_293/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_293/Relu«
dense_294/MatMul/ReadVariableOpReadVariableOp(dense_294_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02!
dense_294/MatMul/ReadVariableOp§
dense_294/MatMulMatMuldense_293/Relu:activations:0'dense_294/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_294/MatMulª
 dense_294/BiasAdd/ReadVariableOpReadVariableOp)dense_294_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_294/BiasAdd/ReadVariableOp©
dense_294/BiasAddBiasAdddense_294/MatMul:product:0(dense_294/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_294/BiasAddv
dense_294/ReluReludense_294/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_294/Relu­
dense_295/MatMul/ReadVariableOpReadVariableOp(dense_295_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02!
dense_295/MatMul/ReadVariableOp©
dense_295/MatMulMatMuldense_294/Relu:activations:0'dense_295/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_295/MatMul¬
 dense_295/BiasAdd/ReadVariableOpReadVariableOp)dense_295_biasadd_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_295/BiasAdd/ReadVariableOp«
dense_295/BiasAddBiasAdddense_295/MatMul:product:0(dense_295/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_295/BiasAdd¿
IdentityIdentitydense_295/BiasAdd:output:0!^dense_293/BiasAdd/ReadVariableOp ^dense_293/MatMul/ReadVariableOp!^dense_294/BiasAdd/ReadVariableOp ^dense_294/MatMul/ReadVariableOp!^dense_295/BiasAdd/ReadVariableOp ^dense_295/MatMul/ReadVariableOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 dense_293/BiasAdd/ReadVariableOp dense_293/BiasAdd/ReadVariableOp2B
dense_293/MatMul/ReadVariableOpdense_293/MatMul/ReadVariableOp2D
 dense_294/BiasAdd/ReadVariableOp dense_294/BiasAdd/ReadVariableOp2B
dense_294/MatMul/ReadVariableOpdense_294/MatMul/ReadVariableOp2D
 dense_295/BiasAdd/ReadVariableOp dense_295/BiasAdd/ReadVariableOp2B
dense_295/MatMul/ReadVariableOpdense_295/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ó
.__inference_decoder_PI_layer_call_fn_146872719

inputs
unknown:d
	unknown_0:d
	unknown_1:	dÄ
	unknown_2:	Ä
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_PI_layer_call_and_return_conditional_losses_1468723722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£G
é
"__inference__traced_save_146872988
file_prefixI
Esavev2_autoencoder_encoder_pi_dense_100_pi_kernel_read_readvariableopG
Csavev2_autoencoder_encoder_pi_dense_100_pi_bias_read_readvariableopF
Bsavev2_autoencoder_encoder_pi_dense_284_kernel_read_readvariableopD
@savev2_autoencoder_encoder_pi_dense_284_bias_read_readvariableopF
Bsavev2_autoencoder_encoder_pi_dense_285_kernel_read_readvariableopD
@savev2_autoencoder_encoder_pi_dense_285_bias_read_readvariableopF
Bsavev2_autoencoder_encoder_pi_dense_286_kernel_read_readvariableopD
@savev2_autoencoder_encoder_pi_dense_286_bias_read_readvariableopF
Bsavev2_autoencoder_decoder_pi_dense_287_kernel_read_readvariableopD
@savev2_autoencoder_decoder_pi_dense_287_bias_read_readvariableopF
Bsavev2_autoencoder_decoder_pi_dense_288_kernel_read_readvariableopD
@savev2_autoencoder_decoder_pi_dense_288_bias_read_readvariableopF
Bsavev2_autoencoder_decoder_pc_dense_289_kernel_read_readvariableopD
@savev2_autoencoder_decoder_pc_dense_289_bias_read_readvariableopF
Bsavev2_autoencoder_decoder_pc_dense_290_kernel_read_readvariableopD
@savev2_autoencoder_decoder_pc_dense_290_bias_read_readvariableopO
Ksavev2_autoencoder_encoder_image_dense_100_image_kernel_read_readvariableopM
Isavev2_autoencoder_encoder_image_dense_100_image_bias_read_readvariableopI
Esavev2_autoencoder_encoder_image_dense_291_kernel_read_readvariableopG
Csavev2_autoencoder_encoder_image_dense_291_bias_read_readvariableopI
Esavev2_autoencoder_encoder_image_dense_292_kernel_read_readvariableopG
Csavev2_autoencoder_encoder_image_dense_292_bias_read_readvariableopI
Esavev2_autoencoder_decoder_image_dense_293_kernel_read_readvariableopG
Csavev2_autoencoder_decoder_image_dense_293_bias_read_readvariableopI
Esavev2_autoencoder_decoder_image_dense_294_kernel_read_readvariableopG
Csavev2_autoencoder_decoder_image_dense_294_bias_read_readvariableopI
Esavev2_autoencoder_decoder_image_dense_295_kernel_read_readvariableopG
Csavev2_autoencoder_decoder_image_dense_295_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename«
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*½
value³B°B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesà
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Esavev2_autoencoder_encoder_pi_dense_100_pi_kernel_read_readvariableopCsavev2_autoencoder_encoder_pi_dense_100_pi_bias_read_readvariableopBsavev2_autoencoder_encoder_pi_dense_284_kernel_read_readvariableop@savev2_autoencoder_encoder_pi_dense_284_bias_read_readvariableopBsavev2_autoencoder_encoder_pi_dense_285_kernel_read_readvariableop@savev2_autoencoder_encoder_pi_dense_285_bias_read_readvariableopBsavev2_autoencoder_encoder_pi_dense_286_kernel_read_readvariableop@savev2_autoencoder_encoder_pi_dense_286_bias_read_readvariableopBsavev2_autoencoder_decoder_pi_dense_287_kernel_read_readvariableop@savev2_autoencoder_decoder_pi_dense_287_bias_read_readvariableopBsavev2_autoencoder_decoder_pi_dense_288_kernel_read_readvariableop@savev2_autoencoder_decoder_pi_dense_288_bias_read_readvariableopBsavev2_autoencoder_decoder_pc_dense_289_kernel_read_readvariableop@savev2_autoencoder_decoder_pc_dense_289_bias_read_readvariableopBsavev2_autoencoder_decoder_pc_dense_290_kernel_read_readvariableop@savev2_autoencoder_decoder_pc_dense_290_bias_read_readvariableopKsavev2_autoencoder_encoder_image_dense_100_image_kernel_read_readvariableopIsavev2_autoencoder_encoder_image_dense_100_image_bias_read_readvariableopEsavev2_autoencoder_encoder_image_dense_291_kernel_read_readvariableopCsavev2_autoencoder_encoder_image_dense_291_bias_read_readvariableopEsavev2_autoencoder_encoder_image_dense_292_kernel_read_readvariableopCsavev2_autoencoder_encoder_image_dense_292_bias_read_readvariableopEsavev2_autoencoder_decoder_image_dense_293_kernel_read_readvariableopCsavev2_autoencoder_decoder_image_dense_293_bias_read_readvariableopEsavev2_autoencoder_decoder_image_dense_294_kernel_read_readvariableopCsavev2_autoencoder_decoder_image_dense_294_bias_read_readvariableopEsavev2_autoencoder_decoder_image_dense_295_kernel_read_readvariableopCsavev2_autoencoder_decoder_image_dense_295_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesð
í: :	Äd:d:d2:2:2::2::d:d:	dÄ:Ä:2:2:2::
d:d:d::d::2:2:2d:d:
d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Äd: 

_output_shapes
:d:$ 

_output_shapes

:d2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::$ 

_output_shapes

:2: 

_output_shapes
::$	 

_output_shapes

:d: 


_output_shapes
:d:%!

_output_shapes
:	dÄ:!

_output_shapes	
:Ä:$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::&"
 
_output_shapes
:
d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:2: 

_output_shapes
:2:$ 

_output_shapes

:2d: 

_output_shapes
:d:&"
 
_output_shapes
:
d:"

_output_shapes

::

_output_shapes
: 
Ü
x
/__inference_sampling_28_layer_call_fn_146872742
inputs_0
inputs_1
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sampling_28_layer_call_and_return_conditional_losses_1468723092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1


/__inference_autoencoder_layer_call_fn_146872489
input_1
input_2
unknown:	Äd
	unknown_0:d
	unknown_1:d2
	unknown_2:2
	unknown_3:2
	unknown_4:
	unknown_5:2
	unknown_6:
	unknown_7:
d
	unknown_8:d
	unknown_9:d

unknown_10:

unknown_11:d

unknown_12:

unknown_13:2

unknown_14:2

unknown_15:2

unknown_16:

unknown_17:d

unknown_18:d

unknown_19:	dÄ

unknown_20:	Ä

unknown_21:2

unknown_22:2

unknown_23:2d

unknown_24:d

unknown_25:
d

unknown_26:

identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *?
_output_shapes-
+:ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ: *>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_autoencoder_layer_call_and_return_conditional_losses_1468724232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:ÿÿÿÿÿÿÿÿÿÄ:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
!
_user_specified_name	input_1:RN
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
Ã

L__inference_decoder_image_layer_call_and_return_conditional_losses_146872406

inputs:
(dense_293_matmul_readvariableop_resource:27
)dense_293_biasadd_readvariableop_resource:2:
(dense_294_matmul_readvariableop_resource:2d7
)dense_294_biasadd_readvariableop_resource:d<
(dense_295_matmul_readvariableop_resource:
d9
)dense_295_biasadd_readvariableop_resource:

identity¢ dense_293/BiasAdd/ReadVariableOp¢dense_293/MatMul/ReadVariableOp¢ dense_294/BiasAdd/ReadVariableOp¢dense_294/MatMul/ReadVariableOp¢ dense_295/BiasAdd/ReadVariableOp¢dense_295/MatMul/ReadVariableOp«
dense_293/MatMul/ReadVariableOpReadVariableOp(dense_293_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02!
dense_293/MatMul/ReadVariableOp
dense_293/MatMulMatMulinputs'dense_293/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_293/MatMulª
 dense_293/BiasAdd/ReadVariableOpReadVariableOp)dense_293_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02"
 dense_293/BiasAdd/ReadVariableOp©
dense_293/BiasAddBiasAdddense_293/MatMul:product:0(dense_293/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_293/BiasAddv
dense_293/ReluReludense_293/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
dense_293/Relu«
dense_294/MatMul/ReadVariableOpReadVariableOp(dense_294_matmul_readvariableop_resource*
_output_shapes

:2d*
dtype02!
dense_294/MatMul/ReadVariableOp§
dense_294/MatMulMatMuldense_293/Relu:activations:0'dense_294/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_294/MatMulª
 dense_294/BiasAdd/ReadVariableOpReadVariableOp)dense_294_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02"
 dense_294/BiasAdd/ReadVariableOp©
dense_294/BiasAddBiasAdddense_294/MatMul:product:0(dense_294/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_294/BiasAddv
dense_294/ReluReludense_294/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_294/Relu­
dense_295/MatMul/ReadVariableOpReadVariableOp(dense_295_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02!
dense_295/MatMul/ReadVariableOp©
dense_295/MatMulMatMuldense_294/Relu:activations:0'dense_295/MatMul/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_295/MatMul¬
 dense_295/BiasAdd/ReadVariableOpReadVariableOp)dense_295_biasadd_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_295/BiasAdd/ReadVariableOp«
dense_295/BiasAddBiasAdddense_295/MatMul:product:0(dense_295/BiasAdd/ReadVariableOp:value:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_295/BiasAdd¿
IdentityIdentitydense_295/BiasAdd:output:0!^dense_293/BiasAdd/ReadVariableOp ^dense_293/MatMul/ReadVariableOp!^dense_294/BiasAdd/ReadVariableOp ^dense_294/MatMul/ReadVariableOp!^dense_295/BiasAdd/ReadVariableOp ^dense_295/MatMul/ReadVariableOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2D
 dense_293/BiasAdd/ReadVariableOp dense_293/BiasAdd/ReadVariableOp2B
dense_293/MatMul/ReadVariableOpdense_293/MatMul/ReadVariableOp2D
 dense_294/BiasAdd/ReadVariableOp dense_294/BiasAdd/ReadVariableOp2B
dense_294/MatMul/ReadVariableOpdense_294/MatMul/ReadVariableOp2D
 dense_295/BiasAdd/ReadVariableOp dense_295/BiasAdd/ReadVariableOp2B
dense_295/MatMul/ReadVariableOpdense_295/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
Ñ
.__inference_decoder_PC_layer_call_fn_146872778

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_decoder_PC_layer_call_and_return_conditional_losses_1468723422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ!
Ã
L__inference_encoder_image_layer_call_and_return_conditional_losses_146872838

inputsB
.dense_100_image_matmul_readvariableop_resource:
d=
/dense_100_image_biasadd_readvariableop_resource:d:
(dense_291_matmul_readvariableop_resource:d7
)dense_291_biasadd_readvariableop_resource::
(dense_292_matmul_readvariableop_resource:d7
)dense_292_biasadd_readvariableop_resource:
identity

identity_1¢&dense_100_image/BiasAdd/ReadVariableOp¢%dense_100_image/MatMul/ReadVariableOp¢ dense_291/BiasAdd/ReadVariableOp¢dense_291/MatMul/ReadVariableOp¢ dense_292/BiasAdd/ReadVariableOp¢dense_292/MatMul/ReadVariableOp¿
%dense_100_image/MatMul/ReadVariableOpReadVariableOp.dense_100_image_matmul_readvariableop_resource* 
_output_shapes
:
d*
dtype02'
%dense_100_image/MatMul/ReadVariableOp£
dense_100_image/MatMulMatMulinputs-dense_100_image/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_image/MatMul¼
&dense_100_image/BiasAdd/ReadVariableOpReadVariableOp/dense_100_image_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02(
&dense_100_image/BiasAdd/ReadVariableOpÁ
dense_100_image/BiasAddBiasAdd dense_100_image/MatMul:product:0.dense_100_image/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_image/BiasAdd
dense_100_image/ReluRelu dense_100_image/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd2
dense_100_image/Relu«
dense_291/MatMul/ReadVariableOpReadVariableOp(dense_291_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_291/MatMul/ReadVariableOp­
dense_291/MatMulMatMul"dense_100_image/Relu:activations:0'dense_291/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_291/MatMulª
 dense_291/BiasAdd/ReadVariableOpReadVariableOp)dense_291_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_291/BiasAdd/ReadVariableOp©
dense_291/BiasAddBiasAdddense_291/MatMul:product:0(dense_291/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_291/BiasAdd«
dense_292/MatMul/ReadVariableOpReadVariableOp(dense_292_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02!
dense_292/MatMul/ReadVariableOp­
dense_292/MatMulMatMul"dense_100_image/Relu:activations:0'dense_292/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_292/MatMulª
 dense_292/BiasAdd/ReadVariableOpReadVariableOp)dense_292_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_292/BiasAdd/ReadVariableOp©
dense_292/BiasAddBiasAdddense_292/MatMul:product:0(dense_292/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_292/BiasAddÉ
IdentityIdentitydense_291/BiasAdd:output:0'^dense_100_image/BiasAdd/ReadVariableOp&^dense_100_image/MatMul/ReadVariableOp!^dense_291/BiasAdd/ReadVariableOp ^dense_291/MatMul/ReadVariableOp!^dense_292/BiasAdd/ReadVariableOp ^dense_292/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÍ

Identity_1Identitydense_292/BiasAdd:output:0'^dense_100_image/BiasAdd/ReadVariableOp&^dense_100_image/MatMul/ReadVariableOp!^dense_291/BiasAdd/ReadVariableOp ^dense_291/MatMul/ReadVariableOp!^dense_292/BiasAdd/ReadVariableOp ^dense_292/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&dense_100_image/BiasAdd/ReadVariableOp&dense_100_image/BiasAdd/ReadVariableOp2N
%dense_100_image/MatMul/ReadVariableOp%dense_100_image/MatMul/ReadVariableOp2D
 dense_291/BiasAdd/ReadVariableOp dense_291/BiasAdd/ReadVariableOp2B
dense_291/MatMul/ReadVariableOpdense_291/MatMul/ReadVariableOp2D
 dense_292/BiasAdd/ReadVariableOp dense_292/BiasAdd/ReadVariableOp2B
dense_292/MatMul/ReadVariableOpdense_292/MatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿÄ
=
input_22
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿ=
output_11
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÄ>
output_22
StatefulPartitionedCall:1ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¢í


encoder_PI

decoder_PI
sampling
shared_decoder
encoder_image
decoder_image
regularization_losses
trainable_variables
		variables

	keras_api

signatures
ï__call__
ð_default_save_signature
+ñ&call_and_return_all_conditional_losses"õ
_tf_keras_modelÛ{"name": "autoencoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "VariationalAutoEncoder", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "__tuple__", "items": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 2500]}, "float32", "input_1"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [1, 16384]}, "float32", "input_2"]}]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "VariationalAutoEncoder"}}
ä
	dense_100
dense_50

dense_mean
	dense_var
regularization_losses
trainable_variables
	variables
	keras_api
ò__call__
+ó&call_and_return_all_conditional_losses"
_tf_keras_layerý{"name": "encoder_PI", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Encoder_PI", "config": {"layer was saved without config": true}}
É
	dense_100
dense_output
regularization_losses
trainable_variables
	variables
	keras_api
ô__call__
+õ&call_and_return_all_conditional_losses"
_tf_keras_layerý{"name": "decoder_PI", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Decoder_PI", "config": {"layer was saved without config": true}}
Ô
regularization_losses
trainable_variables
	variables
	keras_api
ö__call__
+÷&call_and_return_all_conditional_losses"Ã
_tf_keras_layer©{"name": "sampling_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Sampling", "config": {"name": "sampling_28", "trainable": true, "dtype": "float32"}, "shared_object_id": 0}
Ì
dense_20
dense_output
 regularization_losses
!trainable_variables
"	variables
#	keras_api
ø__call__
+ù&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "decoder_PC", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Shared_Decoder", "config": {"layer was saved without config": true}}
Ü
$	dense_100
%
dense_mean
&	dense_var
'regularization_losses
(trainable_variables
)	variables
*	keras_api
ú__call__
+û&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "encoder_image", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Encoder_image", "config": {"layer was saved without config": true}}
Ý
+dense_50
,	dense_100
-dense_output
.regularization_losses
/trainable_variables
0	variables
1	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer{"name": "decoder_image", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Decoder_image", "config": {"layer was saved without config": true}}
 "
trackable_list_wrapper
ö
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27"
trackable_list_wrapper
ö
20
31
42
53
64
75
86
97
:8
;9
<10
=11
>12
?13
@14
A15
B16
C17
D18
E19
F20
G21
H22
I23
J24
K25
L26
M27"
trackable_list_wrapper
Î
regularization_losses
trainable_variables
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
		variables
ï__call__
ð_default_save_signature
+ñ&call_and_return_all_conditional_losses
'ñ"call_and_return_conditional_losses"
_generic_user_object
-
þserving_default"
signature_map
Ú

2kernel
3bias
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses"³
_tf_keras_layer{"name": "dense_100_PI", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_100_PI", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2500}}, "shared_object_id": 4}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2500]}}
Ñ

4kernel
5bias
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense_284", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_284", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 8}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
Ó

6kernel
7bias
[regularization_losses
\trainable_variables
]	variables
^	keras_api
__call__
+&call_and_return_all_conditional_losses"¬
_tf_keras_layer{"name": "dense_285", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_285", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 50]}}
Ô

8kernel
9bias
_regularization_losses
`trainable_variables
a	variables
b	keras_api
__call__
+&call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_286", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_286", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 50]}}
 "
trackable_list_wrapper
X
20
31
42
53
64
75
86
97"
trackable_list_wrapper
X
20
31
42
53
64
75
86
97"
trackable_list_wrapper
°
regularization_losses
trainable_variables
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
	variables
ò__call__
+ó&call_and_return_all_conditional_losses
'ó"call_and_return_conditional_losses"
_generic_user_object
Ò

:kernel
;bias
hregularization_losses
itrainable_variables
j	variables
k	keras_api
__call__
+&call_and_return_all_conditional_losses"«
_tf_keras_layer{"name": "dense_287", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_287", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
Ù

<kernel
=bias
lregularization_losses
mtrainable_variables
n	variables
o	keras_api
__call__
+&call_and_return_all_conditional_losses"²
_tf_keras_layer{"name": "dense_288", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_288", "trainable": true, "dtype": "float32", "units": 2500, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
 "
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
<
:0
;1
<2
=3"
trackable_list_wrapper
°
regularization_losses
trainable_variables
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
ô__call__
+õ&call_and_return_all_conditional_losses
'õ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
regularization_losses
trainable_variables
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
	variables
ö__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
Ñ

>kernel
?bias
zregularization_losses
{trainable_variables
|	variables
}	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense_289", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_289", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
Ö

@kernel
Abias
~regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_290", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_290", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 50]}}
 "
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
µ
 regularization_losses
!trainable_variables
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
"	variables
ø__call__
+ù&call_and_return_all_conditional_losses
'ù"call_and_return_conditional_losses"
_generic_user_object
ê

Bkernel
Cbias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¿
_tf_keras_layer¥{"name": "dense_100_image", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_100_image", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16384}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 16384]}}
Ú

Dkernel
Ebias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¯
_tf_keras_layer{"name": "dense_291", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_291", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
Ú

Fkernel
Gbias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¯
_tf_keras_layer{"name": "dense_292", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_292", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
 "
trackable_list_wrapper
J
B0
C1
D2
E3
F4
G5"
trackable_list_wrapper
J
B0
C1
D2
E3
F4
G5"
trackable_list_wrapper
µ
'regularization_losses
(trainable_variables
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
ú__call__
+û&call_and_return_all_conditional_losses
'û"call_and_return_conditional_losses"
_generic_user_object
Õ

Hkernel
Ibias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ª
_tf_keras_layer{"name": "dense_293", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_293", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
Ø

Jkernel
Kbias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"­
_tf_keras_layer{"name": "dense_294", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_294", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 50]}}
Þ

Lkernel
Mbias
 regularization_losses
¡trainable_variables
¢	variables
£	keras_api
__call__
+&call_and_return_all_conditional_losses"³
_tf_keras_layer{"name": "dense_295", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_295", "trainable": true, "dtype": "float32", "units": 16384, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 100]}}
 "
trackable_list_wrapper
J
H0
I1
J2
K3
L4
M5"
trackable_list_wrapper
J
H0
I1
J2
K3
L4
M5"
trackable_list_wrapper
µ
.regularization_losses
/trainable_variables
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
0	variables
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
=:;	Äd2*autoencoder/encoder_PI/dense_100_PI/kernel
6:4d2(autoencoder/encoder_PI/dense_100_PI/bias
9:7d22'autoencoder/encoder_PI/dense_284/kernel
3:122%autoencoder/encoder_PI/dense_284/bias
9:722'autoencoder/encoder_PI/dense_285/kernel
3:12%autoencoder/encoder_PI/dense_285/bias
9:722'autoencoder/encoder_PI/dense_286/kernel
3:12%autoencoder/encoder_PI/dense_286/bias
9:7d2'autoencoder/decoder_PI/dense_287/kernel
3:1d2%autoencoder/decoder_PI/dense_287/bias
::8	dÄ2'autoencoder/decoder_PI/dense_288/kernel
4:2Ä2%autoencoder/decoder_PI/dense_288/bias
9:722'autoencoder/decoder_PC/dense_289/kernel
3:122%autoencoder/decoder_PC/dense_289/bias
9:722'autoencoder/decoder_PC/dense_290/kernel
3:12%autoencoder/decoder_PC/dense_290/bias
D:B
d20autoencoder/encoder_image/dense_100_image/kernel
<::d2.autoencoder/encoder_image/dense_100_image/bias
<::d2*autoencoder/encoder_image/dense_291/kernel
6:42(autoencoder/encoder_image/dense_291/bias
<::d2*autoencoder/encoder_image/dense_292/kernel
6:42(autoencoder/encoder_image/dense_292/bias
<::22*autoencoder/decoder_image/dense_293/kernel
6:422(autoencoder/decoder_image/dense_293/bias
<::2d2*autoencoder/decoder_image/dense_294/kernel
6:4d2(autoencoder/decoder_image/dense_294/bias
>:<
d2*autoencoder/decoder_image/dense_295/kernel
8:62(autoencoder/decoder_image/dense_295/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
µ
Sregularization_losses
Ttrainable_variables
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
U	variables
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
µ
Wregularization_losses
Xtrainable_variables
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
Y	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
µ
[regularization_losses
\trainable_variables
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
]	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
_regularization_losses
`trainable_variables
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
a	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
µ
hregularization_losses
itrainable_variables
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
j	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
µ
lregularization_losses
mtrainable_variables
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
n	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
µ
zregularization_losses
{trainable_variables
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
|	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
¶
~regularization_losses
trainable_variables
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
¸
regularization_losses
trainable_variables
Ñnon_trainable_variables
Òlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
¸
regularization_losses
trainable_variables
Önon_trainable_variables
×layers
Ømetrics
 Ùlayer_regularization_losses
Úlayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
¸
regularization_losses
trainable_variables
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
¸
regularization_losses
trainable_variables
ànon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
¸
regularization_losses
trainable_variables
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
¸
 regularization_losses
¡trainable_variables
ênon_trainable_variables
ëlayers
ìmetrics
 ílayer_regularization_losses
îlayer_metrics
¢	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¨2¥
/__inference_autoencoder_layer_call_fn_146872489ñ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *Q¢N
L¢I
"
input_1ÿÿÿÿÿÿÿÿÿÄ
# 
input_2ÿÿÿÿÿÿÿÿÿ
2
$__inference__wrapped_model_146872174á
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *Q¢N
L¢I
"
input_1ÿÿÿÿÿÿÿÿÿÄ
# 
input_2ÿÿÿÿÿÿÿÿÿ
Ã2À
J__inference_autoencoder_layer_call_and_return_conditional_losses_146872423ñ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *Q¢N
L¢I
"
input_1ÿÿÿÿÿÿÿÿÿÄ
# 
input_2ÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_encoder_PI_layer_call_fn_146872675¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_encoder_PI_layer_call_and_return_conditional_losses_146872706¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_decoder_PI_layer_call_fn_146872719¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_decoder_PI_layer_call_and_return_conditional_losses_146872736¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_sampling_28_layer_call_fn_146872742¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_sampling_28_layer_call_and_return_conditional_losses_146872765¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_decoder_PC_layer_call_fn_146872778¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_decoder_PC_layer_call_and_return_conditional_losses_146872795¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_encoder_image_layer_call_fn_146872814¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_encoder_image_layer_call_and_return_conditional_losses_146872838¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_decoder_image_layer_call_fn_146872855¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_decoder_image_layer_call_and_return_conditional_losses_146872879¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÕBÒ
'__inference_signature_wrapper_146872652input_1input_2"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
$__inference__wrapped_model_146872174ã23456789BCDEFG>?@A:;<=HIJKLM[¢X
Q¢N
L¢I
"
input_1ÿÿÿÿÿÿÿÿÿÄ
# 
input_2ÿÿÿÿÿÿÿÿÿ
ª "fªc
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿÄ
0
output_2$!
output_2ÿÿÿÿÿÿÿÿÿ¨
J__inference_autoencoder_layer_call_and_return_conditional_losses_146872423Ù23456789BCDEFG>?@A:;<=HIJKLM[¢X
Q¢N
L¢I
"
input_1ÿÿÿÿÿÿÿÿÿÄ
# 
input_2ÿÿÿÿÿÿÿÿÿ
ª "\¢Y
D¢A

0/0ÿÿÿÿÿÿÿÿÿÄ

0/1ÿÿÿÿÿÿÿÿÿ

	
1/0 ñ
/__inference_autoencoder_layer_call_fn_146872489½23456789BCDEFG>?@A:;<=HIJKLM[¢X
Q¢N
L¢I
"
input_1ÿÿÿÿÿÿÿÿÿÄ
# 
input_2ÿÿÿÿÿÿÿÿÿ
ª "@¢=

0ÿÿÿÿÿÿÿÿÿÄ

1ÿÿÿÿÿÿÿÿÿ«
I__inference_decoder_PC_layer_call_and_return_conditional_losses_146872795^>?@A/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_decoder_PC_layer_call_fn_146872778Q>?@A/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
I__inference_decoder_PI_layer_call_and_return_conditional_losses_146872736_:;<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÄ
 
.__inference_decoder_PI_layer_call_fn_146872719R:;<=/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÄ²
L__inference_decoder_image_layer_call_and_return_conditional_losses_146872879bHIJKLM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_decoder_image_layer_call_fn_146872855UHIJKLM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ×
I__inference_encoder_PI_layer_call_and_return_conditional_losses_146872706234567890¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ­
.__inference_encoder_PI_layer_call_fn_146872675{234567890¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÄ
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿÙ
L__inference_encoder_image_layer_call_and_return_conditional_losses_146872838BCDEFG1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "K¢H
A¢>

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ
 ¯
1__inference_encoder_image_layer_call_fn_146872814zBCDEFG1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "=¢:

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿÒ
J__inference_sampling_28_layer_call_and_return_conditional_losses_146872765Z¢W
P¢M
K¢H
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ©
/__inference_sampling_28_layer_call_fn_146872742vZ¢W
P¢M
K¢H
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ 
'__inference_signature_wrapper_146872652ô23456789BCDEFG>?@A:;<=HIJKLMl¢i
¢ 
bª_
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿÄ
.
input_2# 
input_2ÿÿÿÿÿÿÿÿÿ"fªc
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿÄ
0
output_2$!
output_2ÿÿÿÿÿÿÿÿÿ