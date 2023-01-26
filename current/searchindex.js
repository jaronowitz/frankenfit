Search.setIndex({docnames:["api","backends","branching_and_grouping","cover","crossval","dataframes","examples","hyperparams","implementing_transforms","install","synopsis","tips_tricks","transforms_and_pipelines"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.md","backends.md","branching_and_grouping.md","cover.md","crossval.md","dataframes.md","examples.md","hyperparams.md","implementing_transforms.md","install.md","synopsis.md","tips_tricks.md","transforms_and_pipelines.md"],objects:{"":[[0,0,0,"-","frankenfit"]],"frankenfit.DataFramePipeline":[[0,2,1,"","fit"],[0,3,1,"","fit_transform_class"]],"frankenfit.FitTransform":[[0,2,1,"","apply"],[0,2,1,"","bindings"],[0,2,1,"","materialize_state"],[0,4,1,"","name"],[0,2,1,"","on_backend"],[0,2,1,"","resolved_transform"],[0,2,1,"","state"]],"frankenfit.HP":[[0,2,1,"","resolve"],[0,2,1,"","resolve_maybe"]],"frankenfit.HPCols":[[0,2,1,"","maybe_from_value"],[0,2,1,"","resolve"]],"frankenfit.HPDict":[[0,2,1,"","resolve"]],"frankenfit.HPFmtStr":[[0,2,1,"","resolve"]],"frankenfit.HPLambda":[[0,2,1,"","resolve"]],"frankenfit.Pipeline":[[0,2,1,"","apply"],[0,2,1,"","apply_fit_transform"],[0,2,1,"","if_fitting"],[0,2,1,"","then"],[0,3,1,"","transforms"],[0,2,1,"","with_methods"]],"frankenfit.StatelessTransform":[[0,2,1,"","apply"]],"frankenfit.Transform":[[0,2,1,"","_apply"],[0,2,1,"","_fit"],[0,3,1,"","backend"],[0,2,1,"","fit"],[0,3,1,"","fit_transform_class"],[0,2,1,"","hyperparams"],[0,4,1,"","name"],[0,2,1,"","on_backend"],[0,2,1,"","parallel_backend"],[0,2,1,"","params"],[0,3,1,"","pure"],[0,2,1,"","resolve"],[0,3,1,"","tag"],[0,2,1,"","visualize"]],"frankenfit.UniversalPipeline":[[0,2,1,"","fit"],[0,3,1,"","fit_transform_class"]],"frankenfit.backend":[[0,1,1,"","DaskFuture"]],"frankenfit.core":[[0,1,1,"","ApplyFitTransform"],[0,7,1,"","DEFAULT_VISUALIZE_DIGRAPH_KWARGS"],[0,1,1,"","IfPipelineIsFitting"],[0,1,1,"","LocalFuture"],[0,1,1,"","PipelineMember"]],"frankenfit.core.PipelineMember":[[0,2,1,"","__add__"],[0,2,1,"","_children"],[0,2,1,"","_visualize"],[0,2,1,"","find_by_name"],[0,4,1,"","name"],[0,2,1,"","then"]],"frankenfit.dataframe":[[0,1,1,"","Affix"],[0,1,1,"","Assign"],[0,1,1,"","Clip"],[0,1,1,"","Copy"],[0,1,1,"","Correlation"],[0,1,1,"","DataFrameCallChain"],[0,1,1,"","DataFramePipelineInterface"],[0,1,1,"","DeMean"],[0,1,1,"","Drop"],[0,1,1,"","Filter"],[0,1,1,"","GroupByCols"],[0,1,1,"","ImputeConstant"],[0,1,1,"","ImputeMean"],[0,1,1,"","Join"],[0,1,1,"","Pipe"],[0,1,1,"","Prefix"],[0,1,1,"","ReadDataFrame"],[0,1,1,"","ReadDataset"],[0,1,1,"","ReadPandasCSV"],[0,1,1,"","Rename"],[0,1,1,"","SKLearn"],[0,1,1,"","Select"],[0,1,1,"","Statsmodels"],[0,1,1,"","Suffix"],[0,1,1,"","Winsorize"],[0,1,1,"","WriteDataset"],[0,1,1,"","WritePandasCSV"],[0,1,1,"","ZScore"]],"frankenfit.dataframe.DataFrameCallChain":[[0,2,1,"","affix"],[0,2,1,"","assign"],[0,2,1,"","clip"],[0,2,1,"","copy"],[0,2,1,"","correlation"],[0,2,1,"","de_mean"],[0,2,1,"","drop"],[0,2,1,"","filter"],[0,2,1,"","impute_constant"],[0,2,1,"","impute_mean"],[0,2,1,"","pipe"],[0,2,1,"","prefix"],[0,2,1,"","read_data_frame"],[0,2,1,"","read_dataset"],[0,2,1,"","read_pandas_csv"],[0,2,1,"","rename"],[0,2,1,"","select"],[0,2,1,"","sk_learn"],[0,2,1,"","statsmodels"],[0,2,1,"","suffix"],[0,2,1,"","winsorize"],[0,2,1,"","write_dataset"],[0,2,1,"","write_pandas_csv"],[0,2,1,"","z_score"]],"frankenfit.dataframe.DataFramePipelineInterface":[[0,2,1,"","group_by_cols"],[0,2,1,"","join"]],"frankenfit.universal":[[0,1,1,"","ForBindings"],[0,1,1,"","Identity"],[0,1,1,"","IfFittingDataHasProperty"],[0,1,1,"","IfHyperparamIsTrue"],[0,1,1,"","IfHyperparamLambda"],[0,1,1,"","LogMessage"],[0,1,1,"","Print"],[0,1,1,"","StateOf"],[0,1,1,"","StatefulLambda"],[0,1,1,"","StatelessLambda"],[0,1,1,"","UniversalCallChain"],[0,1,1,"","UniversalPipelineInterface"]],"frankenfit.universal.UniversalCallChain":[[0,2,1,"","identity"],[0,2,1,"","if_fitting_data_has_property"],[0,2,1,"","if_hyperparam_is_true"],[0,2,1,"","if_hyperparam_lambda"],[0,2,1,"","log_message"],[0,2,1,"","print"],[0,2,1,"","stateful_lambda"],[0,2,1,"","stateless_lambda"]],"frankenfit.universal.UniversalPipelineInterface":[[0,2,1,"","for_bindings"],[0,2,1,"","last_state"]],frankenfit:[[0,1,1,"","Backend"],[0,1,1,"","ConstantTransform"],[0,1,1,"","DaskBackend"],[0,1,1,"","DataFramePipeline"],[0,1,1,"","FitTransform"],[0,1,1,"","Future"],[0,1,1,"","HP"],[0,1,1,"","HPCols"],[0,1,1,"","HPDict"],[0,1,1,"","HPFmtStr"],[0,1,1,"","HPLambda"],[0,1,1,"","LocalBackend"],[0,1,1,"","NonInitialConstantTransformWarning"],[0,1,1,"","Pipeline"],[0,1,1,"","StatelessTransform"],[0,1,1,"","Transform"],[0,1,1,"","UniversalPipeline"],[0,5,1,"","UnresolvedHyperparameterError"],[0,6,1,"","columns_field"],[0,6,1,"","dict_field"],[0,6,1,"","fmt_str_field"],[0,6,1,"","params"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","exception","Python exception"],"6":["py","function","Python function"],"7":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:exception","6":"py:function","7":"py:data"},terms:{"0":[0,10,12],"00":10,"000000":[10,12],"004":12,"01":[0,12],"025289":0,"029341":0,"03150901":12,"03466946":12,"035128":12,"036225":10,"0463":12,"05":[0,10,12],"050000":12,"060478":10,"07":10,"088":12,"094091":12,"094924":0,"097934":12,"0x7fefecbca020":12,"0x7fefecbcb7c0":12,"0x7fefecbcbac0":12,"0x7fefecd89e40":12,"0x7ff024940250":12,"1":[0,10,12],"10":[0,12],"100":[10,12],"1000":10,"100000":12,"102":12,"103":12,"104":12,"10460":10,"105":12,"10579":12,"106":12,"106885":12,"107":12,"108":12,"109":12,"11":12,"110":12,"111":12,"112":12,"113":12,"114":12,"119":12,"120":12,"121":12,"122":12,"12291":10,"123":12,"124":12,"125":12,"126":12,"12622":10,"127":12,"1273":12,"130000":12,"1337420":12,"14051":10,"1455":12,"146":12,"146558":10,"147":12,"148":12,"149":12,"149891":10,"150":12,"150929":0,"151":12,"155816":12,"160023":10,"162":12,"163":12,"164":12,"165":12,"166":12,"166911":12,"167":12,"168":12,"169":12,"171":12,"183760":0,"1862":3,"18823":12,"190866":12,"195344":0,"19838":10,"199730":12,"2":[0,10,12],"20":[0,10],"200000":12,"2023":3,"204545":12,"206":12,"206467":10,"207":12,"208":12,"209":12,"21":[10,12],"210":12,"211240":12,"212":12,"213":12,"21478":12,"215122":10,"221777":12,"23":[10,12],"23038":12,"235089":12,"2376944":12,"2398":12,"24":12,"24128":10,"243235":12,"24391":10,"249405":12,"25":12,"2537":12,"254154":10,"255136":12,"255490":10,"26970":12,"277632":10,"28452":10,"284856":10,"29":[10,12],"3":[10,12],"30":10,"300786":12,"303313":12,"304":12,"31":[10,12],"315973":12,"326":[10,12],"327":[10,12],"33":10,"332419":12,"334":[10,12],"335":[10,12],"336":12,"34":10,"3462":10,"35":10,"3597":12,"3598":12,"3599":12,"3600":12,"3605":12,"3606":12,"3608":12,"3784":10,"38":10,"386357":12,"39":10,"392375":10,"3932":12,"3934":12,"3982":12,"4":[10,12],"400000":12,"4038":12,"4039":12,"4046":12,"4047":12,"41":10,"416704":10,"42":10,"43":[10,12],"436838":10,"437325":12,"438384":12,"44":12,"4412":10,"448889":12,"45":10,"456":12,"457":12,"457184":12,"45735":12,"457683":12,"458":12,"459534":12,"46890":12,"472800":10,"473374":12,"4781":10,"480529":12,"484358":12,"48794":12,"488533":12,"496":12,"5":[0,10,12],"50":12,"500000":12,"503841":12,"504049224":12,"50794":12,"508533":12,"510258":10,"5346":12,"542317":12,"542816":12,"5429":12,"5463":12,"55":[10,12],"550530":10,"550595":12,"558533":12,"56":[10,12],"560590":12,"5629":12,"56794":12,"568533":12,"57":[10,12],"574508":12,"5774":12,"5775":12,"5776":12,"5784":12,"58":[10,12],"58794":12,"588533":12,"59":[10,12],"596":12,"6":[10,12],"600651":12,"61":[10,12],"62":[10,12],"6229":12,"63":[10,12],"6314941":12,"64":12,"640034":12,"6429":12,"6463":12,"65":[10,12],"650595":12,"669140":12,"67":10,"671":10,"672944":10,"6734":12,"674512":12,"679655":10,"683244":10,"6841":12,"687156":10,"69":10,"697706":12,"7":[10,12],"700000":12,"703098":10,"705":12,"710":12,"710795":10,"711":12,"711384":12,"718":12,"720118":12,"749405":12,"75":[10,12],"753700":12,"76":10,"762":12,"767569":10,"768417":10,"788587529099264":12,"789960":12,"79":12,"793014":12,"797":10,"797287":12,"797940":12,"798533":12,"799722":12,"8":[10,12],"800000":12,"812386":10,"814131":12,"817111":12,"822812":12,"824":12,"83":10,"836491":12,"84":10,"8401":10,"843333":10,"849405":12,"87":12,"877597":12,"889033":10,"89":10,"9":[10,12],"90":12,"902165627":12,"903625":12,"904":12,"906518":10,"91":12,"912":12,"919174":12,"919813":12,"92":12,"920283":10,"921591":0,"925196":12,"925868":10,"926416":12,"926595":12,"927241":12,"929112":12,"93":12,"948906":12,"949405":12,"95":[10,12],"951175":12,"9537":12,"95536379":12,"956":12,"96":12,"969":12,"97":12,"971":12,"972":12,"973":12,"974":12,"98":[10,12],"99":12,"993252":10,"abstract":0,"break":[0,12],"byte":12,"case":[0,8,10,12],"catch":11,"class":[8,11,12],"default":[0,12],"do":[0,9,10,12],"final":12,"float":0,"function":[0,11,12],"import":[0,3,8,10,12],"int":0,"long":[3,12],"new":[0,8,12],"null":[0,12],"public":[0,9],"return":[0,8,11,12],"short":[0,10],"static":0,"true":[0,10,12],"try":12,"while":[9,12],A:[0,3,10,12],AND:3,AS:3,And:12,As:[0,8,10,12],At:[0,12],BE:3,BUT:3,BY:3,But:[0,12],By:[0,12],FOR:3,For:[0,3,9,10,12],IF:[3,10],IN:3,IS:3,If:[0,9,10,12],In:[0,12],It:[0,9,10,11,12],Its:0,NO:3,NOT:3,No:0,Not:0,OF:3,ON:3,OR:3,Of:12,One:[0,12],Or:[0,12],SUCH:3,THE:3,TO:3,The:[3,8,9,10,11],There:12,These:[0,12],To:[9,12],With:[0,9,10],_:12,__add__:[0,12],__call__:12,__getitem__:0,__name__:12,_appli:0,_base:12,_children:0,_comput:0,_description_:0,_dmn:0,_fea:10,_fit:0,_jupyter_mimetyp:12,_pipe_futur:12,_pipe_legaci:12,_pipe_lin:12,_pipe_lines_str:12,_repr_image_svg_xml:12,_repr_mimebundle_:12,_self:0,_summary_:0,_tool:12,_visual:[0,8],_x:0,_y:0,abbrevi:12,abil:10,abl:[0,12],about:[0,12],abov:[0,3,12],accept:[0,12],access:[0,12],accomplish:12,accord:0,achiev:10,acquir:3,act:[0,12],action:12,activ:9,actual:[0,11,12],ad:3,add:[0,9,12],addit:[0,3,12],addition:12,addr:0,address:12,advanc:9,advantag:9,advis:3,affect:0,affix:0,after:[0,12],again:0,against:0,agnost:10,alia:0,all:[0,9,10,11,12],all_col:[0,12],allow:[0,12],allow_unresolv:0,almost:12,alon:3,along:12,alongsid:12,alpha:12,alread:0,alreadi:[0,3,9,12],also:[0,9,10,12],altern:0,although:12,altogeth:12,alwai:[0,12],among:0,an:[0,11,12],analysi:9,analyz:12,ani:[0,3,10,12],annot:9,anoth:[0,5,10,12],anticip:0,anyth:0,api:[3,9,10],appear:0,append:[0,12],applend:0,appli:[0,3,9,11],applic:[0,12],apply_fit_transform:[0,12],apply_fun:[0,12],apply_msg:0,applyfittransform:[0,12],applyresult:0,appropri:0,apt:9,ar:[0,3,9,10,11,12],arbitrari:[0,12],arg:[0,12],argument:[0,12],aris:3,arrai:[0,12],art:3,as_index:0,assert:[0,12],assig:0,assign:[0,8,10,12],assist:0,associ:12,assum:0,assumpt:[0,12],astyp:10,attempt:0,attr:0,attrbut:0,attribut:[0,12],aureliu:3,author:[0,8,9,10],authorship:3,auto:11,autocomplet:9,automat:[0,9,12],avail:[0,9,12],avoid:[0,12],b:3,back:[0,10,12],backend:[3,8,9,12],bake:3,bake_featur:0,baker:3,balanc:0,bane:3,bar:[0,12],bare:[0,12],base:[10,11,12],base_dir:0,basic:12,batch:10,beauti:3,becaus:[0,12],becom:12,been:[0,8,12],befor:[0,12],begin:12,behavior:[5,12],being:[0,9,12],belong:0,below:[9,12],benefit:11,bespok:12,best:12,beta:[11,12],between:[0,10,12],beyond:0,bg_fg:0,bias:12,binari:3,bind:[0,10,12],bindings_sequ:0,bit:12,black:0,block:[10,12],bool:0,both:[0,12],bottom:12,bound:0,box:0,bracket:11,branch:[3,10,12],bread:3,breakpoint:11,breviti:[0,10],brief:0,broadli:0,build:[10,12],built:[0,11,12],bunch:12,busi:3,c:[0,8,12],cach:0,call:[0,10,11],callabl:0,callchain:10,calledprocesserror:12,caller:0,camelcas:12,can:[0,9,10,11,12],cannot:[0,12],capabl:[9,12],captur:0,capture_output:12,carat:[0,10,12],carat_fea:10,care:0,carri:12,categori:12,caus:[0,3],certain:[3,12],chain:[0,11],chall:12,chang:0,charg:3,check:[0,11],check_returncod:12,child:[0,10],children:0,chosen:0,circumst:12,claim:3,clariti:[0,10],class_param:[0,10,12],classmethod:0,clean:[0,12],clearli:12,client:0,clip:[0,10],clunki:11,cmd:12,code:[0,3,9,11],codec:12,coef_:12,coeffici:0,col1:0,col2:0,col:[0,8,11,12],collect:0,color:10,column:[0,8,10,12],columns_field:[0,12],columnstransform:0,com:3,combin:[0,3,10,12],combine_fun:0,combined_model:12,come:12,command:12,common:[0,10,12],compar:0,complet:[0,11,12],completedprocess:12,complex:[0,10,11,12],complic:12,compos:[10,11],composit:12,compris:12,comput:[3,9,11,12],concaten:0,conceptu:12,concis:[10,12],concret:0,condit:3,consequenti:3,consid:[0,12],consist:[0,12],constant:[0,12],constantdataframetransform:0,constanttransform:0,constitu:[10,12],construct:[0,12],constructor:[0,12],consult:0,consum:0,contain:[0,12],content:0,context:0,continu:12,contract:[0,3],contrari:3,contribut:3,contributor:3,control:12,conveni:[0,12],convent:[0,10,12],convert:12,copi:[0,11,12],copyright:3,core:[10,12],corr:[0,12],correl:[0,10,12],correland:0,correspond:[0,12],could:[0,10,12],count:12,cours:12,covari:12,cover:12,creat:[0,11,12],creation:0,cross:[0,3,10],crucial:12,csv:10,current:0,custom:[0,12],cut:[0,10],d:[0,10,12],damag:3,dask:9,dask_futur:0,daskbackend:0,daskfutur:0,data:[0,3,5,12],data_appli:[0,8],data_fit:[0,8],datafram:[3,8,9,10,12],dataframecallchain:0,dataframegroup:0,dataframepipelin:[0,3,10,11,12],dataframepipelineinterfac:0,dataframetransform:0,dataset:[0,10,12],dataset_kwarg:0,datatyp:0,de:[0,8,12],de_mean:[0,12],decid:9,declar:0,decor:12,def:[0,8,12],default_visualize_digraph_kwarg:0,defer:[0,12],defin:[0,10,12],degrad:12,delete_match:0,demean:[0,8,12],depend:[0,9],deprec:12,deprecate_positional_arg:12,depth:[0,10,12],depth_fea:10,deriv:12,describ:[10,12],descript:10,design:[0,9],desir:[0,3],dest:0,dest_col:0,destin:0,detail:12,determin:[0,12],deviat:[0,12],devis:12,df:[0,10,12],df_appli:0,df_fit:0,df_in:10,df_oo:12,df_out:10,diamet:12,diamond:[0,10,12],diamond_model:10,diamonds_df:[0,12],dict:[0,10],dict_field:0,dictcomp:12,dictionari:0,did:12,differ:[0,10,12],difficult:12,digraph:[0,12],digraph_kwarg:0,direct:[3,12],directli:[0,12],discard:0,discuss:12,disjoint:12,displai:12,distinct:[0,12],distort:12,distribut:[0,3,9,12],divid:[10,12],dmn:12,dmn_2col:12,dmn_price_weight:12,doc:12,docstr:0,document:[0,12],doe:[0,12],doesn:12,dollar:10,domain:[0,10,12],don:[0,12],dot:12,dot_command:12,doubt:12,doubtlessli:12,drop:[0,12],dtype:12,duplic:12,dure:12,e:[0,10,12],each:[0,3,10,12],eampl:11,earlier:12,easi:10,easili:12,eat:3,edge_attr:0,effect:[0,12],effici:[0,12],either:0,element:0,els:[0,12],emb:12,embed:[0,12],emit:0,emoji:0,empti:[0,5,12],empty_pipelin:12,enabl:[0,12],encapsul:[10,12],encod:12,end:12,engin:12,enhanc:11,ensur:[0,12],entir:[0,10,12],environ:[0,9],equal:[0,12],equival:[0,12],error:[11,12],especi:12,essenti:12,estim:[10,12],estoppel:3,etc:0,eval_df:12,eval_oos_df:12,evalu:[11,12],even:[0,3,11,12],event:3,everi:[0,12],everyth:9,exact:12,exactli:[0,12],exampl:[0,3,9,10,11,12],except:[0,3,12],excit:3,exclud:12,exclus:3,execut:[0,9,10,12],exemplari:3,exist:12,existing_data_behavior:0,exit:12,exp_price_hat:12,exp_price_hat_fit:12,expect:[0,12],explicit:[0,12],explicitli:0,exploit:10,expm1:[10,12],exponenti:12,expos:0,express:[0,3,11],expressli:3,extend:0,extens:[9,12],extra:9,extract:11,extraordinari:12,extrem:12,f:12,facet:12,fact:[0,12],factori:12,fail:0,failur:3,fals:0,famili:12,far:12,fashion:3,featur:[10,12],feed:[10,12],few:12,fewer:0,ff:[0,8,9,10,11,12],field:[0,12],figsiz:[10,12],figur:12,file:[0,5,12],filepath:0,filter:[0,12],filter_fun:0,find:0,find_by_nam:[0,12],find_by_tag:0,first:[0,12],fit:[0,3],fit_diamond_model:10,fit_dmn:12,fit_fun:0,fit_group_on_all_other_group:10,fit_group_on_self:0,fit_intercept:[0,10,12],fit_model:12,fit_msg:0,fit_regress:12,fit_transform:0,fit_transform_class:0,fitdataframetransform:[0,12],fitting_schedul:[0,10],fittransform:[0,10],fituniversaltransform:0,fitzscor:0,fix:12,flag:0,float64:12,fmt_str_field:0,fo:0,focu:10,fold:10,follow:[0,3,12],fontnam:0,fontsiz:0,foo:[0,12],foobar:0,for_bind:0,forbind:0,form:3,formal:12,format:[0,12],formatt:12,former:12,found:0,four:12,frac:12,frame:[0,5],frankefit:12,frankenfit:[8,10,11,12],free:[0,3,11],freeli:0,frequenc:0,from:[0,3,9,10,11,12],full:0,fulli:0,fun:0,func:12,fundament:10,furthermor:[0,12],g:[0,10,12],g_co:0,gener:[0,10,11,12],general:0,georg:3,get:[0,3,12],get_real_method:12,getattr:12,getlogg:0,github:3,give:[10,12],given:[0,12],go:9,good:[3,10,12],grant:3,graph:12,graphviz:[0,12],greater:[0,12],ground:12,group:[0,3,10,12],group_by_col:[0,10],group_kei:[0,10],groupbycol:0,grouper:0,ha:[0,12],half:12,hand:[0,12],handi:12,handl:12,happen:12,hat_col:[0,10,12],have:[0,3,8,9,12],head:[10,12],heavyweight:12,height:0,henc:[0,12],here:[0,10,12],hereaft:3,herebi:3,herself:0,heterogen:[0,12],high:[10,12],highli:9,him:12,hist:[10,12],hold:[0,12],holder:3,home:10,homepag:3,hood:12,hostedtoolcach:12,how:[0,12],howev:[0,3,9,12],hp:0,hp_name:0,hpcol:[0,11],hpdict:0,hpfmtstr:[0,12],hplambda:0,http:3,hyerparam:8,hypeparamet:0,hyperparam:[0,11],hyperparamet:[3,12],i:[0,10],id:11,ideal:10,ident:[0,11],identifi:[0,12],idiomat:[0,12],if_fit:[0,12],if_fitting_data_has_properti:0,if_hyperparam_is_tru:0,if_hyperparam_lambda:0,iffittingdatahasproperti:0,ifhyperparamistru:0,ifhyperparamlambda:0,ifpipelineisfit:0,ignor:0,illustr:12,imag:10,imagin:12,immedi:12,implement:[0,3,10,12],implementatino:0,impli:3,implic:3,implicit:12,impute_const:[0,10],impute_mean:0,imputeconst:[0,12],imputemean:0,incident:[3,12],includ:[0,3,10],inconveni:12,incorpor:12,increment:12,indent:0,independ:[0,12],index:[0,8,9,10,11,12],index_col:0,index_label:0,indic:[0,12],indirect:3,individu:12,ineffici:12,influenc:0,info:0,infring:3,ing:0,inher:10,inherit:12,initi:[0,10,11,12],inner:0,input:[0,10,12],input_encod:12,input_lin:12,inspect:12,inspir:[10,12],instal:[3,12],instanc:[0,3,12],instanti:[0,12],instead:[0,12],int32:10,intend:10,intent:0,interact:[0,9],intercept:0,intercept_:12,interepret:0,interest:12,intern:12,interpret:11,interrupt:3,introduc:12,invent:12,invok:[0,12],involv:12,ipython:12,irrevoc:3,item:12,iter:0,its:[0,9,10,12],itself:[0,10],j:10,job:12,join:[0,12],jupyt:9,jupyter_integr:12,jupyterintegr:12,jupyterlab:9,just:[0,9,12],kdot:12,keep:0,keep_larg:12,keepcolumn:0,kei:0,kendal:0,kept:12,keyerror:0,keyword:[0,12],kind:[0,11,12],know:[9,12],known:[0,11,12],kwarg:[0,12],labori:12,lambda:[0,10,12],languag:[0,10,12],larger:12,largest:12,last:[0,12],last_stat:0,later:[9,12],latter:12,layer:[0,12],layout:12,lead:0,learn:[0,10,12],least:[0,12],left:0,left_col:0,left_on:0,len:[0,10],length:[0,12],less:0,let:12,level:[0,11],liabil:3,liabl:3,lib:12,librari:[10,11,12],light:12,lightweight:10,like:[0,9,10,11,12],likewis:0,limit:[0,3,12],linear:[10,12],linear_model:[10,12],linearregress:[0,10,12],link:0,list:[0,3,8,12],liter:0,littl:[0,12],ll:12,load:[11,12],loc:12,local:12,localbackend:0,localfutur:0,log1p:[0,10,12],log:[0,10,12],log_messag:0,log_pric:12,log_price_cara_fit:12,log_price_carat:12,log_price_carat_fit:12,log_price_fit:12,log_price_lambda:12,logger:0,logger_nam:0,logic:[0,11,12],logmessag:0,look:12,lookup:12,loss:3,low:10,lower:[0,10],made:[0,3],magic:0,mai:[0,10,12],main:[0,11,12],make:[0,3,11,12],manag:[0,9],mani:[0,10,12],manipul:12,manner:[3,12],manual:12,map:0,marcu:3,mark:0,match:12,materi:3,materialize_st:0,max:[0,3,12],maxban:3,maybe_from_valu:0,mean:[0,8,12],meant:[0,12],medit:3,memori:11,mention:12,merchant:3,mere:12,messag:0,met:3,method:[0,8,9,11,12],method_nam:12,meticul:11,might:[9,12],mime_typ:12,mimebundleformatt:12,mimetyp:12,min:[0,12],min_ob:0,mind:0,minimum:0,miniscul:12,miss:[0,12],mix:12,mode:0,model:[0,10,11,12],modif:3,modifi:12,modul:[0,12],moment:12,monospac:0,more:[0,9,10,12],most:[0,8,12],much:[9,12],multi:0,multipl:12,must:[0,3,8,12],mutat:0,my_model:0,my_pipelin:[0,12],mycustomtransform:0,mygreattransform:12,mypi:[0,11],n:12,naiv:12,name:[0,8,10,12],nameerror:0,namespac:[0,12],nan:0,ndarrai:0,neato_no_op:12,necessari:0,necessarili:3,need:[0,9,12],neglig:3,nest:12,never:[0,12],next:[0,9,12],ngroup:10,no_cach:0,node_attr:0,non:[0,3,12],nonc:0,none:[0,12],noninitialconstanttransformwarn:0,nor:12,notat:12,note:[0,12],noth:[0,12],notic:[3,12],notimplementederror:0,now:[0,12],np:[0,10,12],nperhap:12,nuanc:12,number:[0,12],numer:0,numpi:[0,10,12],obj:[0,12],object:[0,9,10,12],oblig:11,observ:[0,10,12],obtain:[0,10],occur:[0,12],offer:[3,9],often:[0,12],old:0,omit:[0,12],on_backend:0,onc:[0,10,12],one:[0,9,10],ones:0,onli:[0,3,10,12],onto:[0,12],oo:12,op:0,open:[0,3],oper:[0,9,10,12],opt:12,option:[0,12],order:[0,12],ordinari:12,ordinarili:[0,12],orer:0,org:3,organ:12,origin:12,other:[0,3,10],other_pipelin:0,otherwis:[0,3,12],ought:0,our:12,out:[0,3,10,12],outer:0,outlier:[0,10,12],output:[0,10,12],outsid:[0,12],over:12,overrid:[0,8],overridden:12,overview:[0,3],overwrit:0,own:[0,3,9,11,12],p1:12,p2:12,p:0,p_co:0,packag:[9,12],pair:0,panda:[0,8,9,10,12],parallel:[3,8,10],parallel_backend:0,param:[0,8,11,12],paramet:0,parameter:0,parent:0,parquet:0,part:[0,3,12],parti:10,particular:[0,3,12],particularli:[0,12],partitioning_schema:0,pass:[0,12],path:[0,9],pavilion:12,pd:[0,8],pearson:0,peculiar:3,per:0,percent:0,percentil:0,perform:[10,12],perhap:12,permit:3,perpetu:3,perspect:0,pick:12,pip:[0,9,12],pipe:[0,10,12],pipe_lines_str:12,pipelin:[3,5,9],pipelinegroup:0,pipelinememb:0,piplin:12,plan:12,pleas:0,plot:[10,12],plugin:[0,12],point:0,pollut:0,posit:12,position:12,posixpath:12,possibl:[0,3,10,12],potenti:[0,12],power:12,practic:12,preced:[0,10],precis:0,predict:[0,10,12],predict_pric:12,predict_price_tag:12,predictor:[0,12],prefer:[0,12],prefix:0,premium:10,prepar:[0,10,12],prepare_featur:12,prepare_training_respons:12,prepend:0,presenc:12,preserv:12,preservs:12,pretti:12,prevent:12,previou:12,previous:12,price:[0,10,12],price_dmn2:0,price_hat:[0,10,12],price_hat_dollar:[10,12],price_model:12,price_model_corr:12,price_model_corr_fit:12,price_orig:[0,12],price_rank:0,price_regress:12,price_scal:0,price_train:[0,10,12],price_win2:0,price_win:0,principl:12,print:0,print_method:12,privileg:12,problem:12,proc:12,proce:[9,12],procedur:12,process:12,procur:3,produc:[0,12],product:[0,11],profit:3,project:[0,3],prone:12,properti:0,provid:[0,3,8,10,12],pull:[0,12],pure:0,purpos:[3,12],put:12,py:12,pyarrow:[0,11],pydataset:[0,10,12],pydoc:9,pyproject:0,python3:12,python:[9,10,12],quantil:12,queri:12,question:[10,12],quiet:12,quit:12,r:[0,12],r_co:0,rais:[0,12],random:[0,10,12],random_st:12,randomli:10,rang:[11,12],rare:0,rather:[0,12],raw:12,re:[0,12],read:[0,5,12],read_csv_arg:0,read_data_fram:0,read_dataset:[0,11],read_pandas_csv:[0,10],readabl:[10,12],readdatafram:0,readdataset:0,reader:[0,11,12],readi:12,readpandascsv:0,real:12,reason:[0,12],recal:12,receiv:[3,12],recent:12,recommend:[0,9,10],recomput:0,recurs:0,redistribut:3,reduc:12,refer:[3,12],referenc:0,regardless:12,regist:12,regress:[0,10,11,12],regress_fit:12,relat:0,reli:0,rememb:12,remov:0,renam:0,render:[0,12],replac:[0,12],repo:10,repr:12,repres:[0,12],reproduc:3,requir:[0,12],resampl:10,research:0,reset_index:10,resolut:[0,12],resolv:0,resolve_fun:0,resolve_mayb:0,resolved_transform:[0,12],respect:[0,8,10,12],respons:[0,10,11,12],response_col:[0,10,12],restrict:0,result:[0,9,10,11,12],retain:3,retriev:12,returncod:12,reusabl:10,revisit:12,rewritten:12,right:[0,3],right_col:0,right_on:0,rogu:12,row:[0,10,12],royalti:3,run:[0,12],run_check:12,runner:10,runtim:12,runtimewarn:0,s:[0,3,9,10,11,12],safe:0,sai:[0,12],sake:[0,10],same:[0,12],sampl:[0,10,12],sample_weight:0,satisfi:3,saw:12,scalar:0,scale:12,scanner_kwarg:0,scatter:[10,12],schedul:0,schema:12,scikit:[0,10,12],score:[0,10,11,12],score_corr:12,score_ms:12,score_predict:12,scratch:0,screen:0,screenshot:[9,11],se:0,search:[3,10,12],section:[9,12],see:[0,9,12],seem:12,select:0,self:[0,8,12],selfdpi:0,selfupi:0,sell:3,send:12,sens:[11,12],separ:[0,12],sequenc:[0,10,12],sequenti:[0,12],seri:[0,8,10,12],serv:[0,12],servic:3,set:[0,9,10,12],setup:12,sever:12,shall:3,shallow:0,shape:[0,12],should:[0,11,12],show:[9,12],si1:10,si2:10,side:[0,12],signatur:[0,9,12],similar:12,similarli:0,simpl:[0,12],simplest:[9,12],simpli:0,simplifi:0,sinc:[0,12],singl:[0,12],site:12,situat:[0,12],size:[10,12],sk_learn:[0,10,12],skip:[9,12],sklearn:[0,10,12],sklearn_class:[0,12],sleek:11,slice:0,sm_class:0,small:0,smaller:12,snake_cas:12,so:[0,9,11,12],softwar:3,some:[0,3,8,10,11,12],someth:12,somewher:[0,12],sooner:11,sort:0,sourc:[0,3,12],speak:[0,12],spearman:0,special:[3,12],specif:[0,10,12],specifi:[0,12],split:3,sqrt:[0,12],squar:12,stacklevel:12,standard:[0,10,12],star:0,start:[0,3,12],state:[0,3,8,10,12],stateful_lambda:0,statefullambda:0,stateless:0,stateless_lambda:[0,12],statelessdataframetransform:0,statelesslambda:[0,12],statelesstransform:[0,12],statement:0,stateof:0,statist:10,statlesstransform:12,statsmodel:[0,10],statu:12,std:12,stderr:12,stdin:12,stdout:[0,12],step:12,still:[0,12],stock:9,store:[0,12],str:[0,8,11,12],strict:3,strictli:12,strike:0,string:[0,12],structur:12,studio:9,style:[0,10,12],stylist:[0,10],sub:[0,11,12],subclass:12,subclass_nam:0,subject:3,submodul:12,subprocess:12,subsampl:12,subset:[0,12],substitut:3,subtract:12,success:12,succinctli:12,sudo:9,suffix:[0,10,11],sugar:[0,12],suggest:[9,12],summar:12,summari:11,suppli:[0,12],support:12,supported_numb:12,suppos:[0,10,12],surfac:3,svg:12,svg_encod:12,symbol:0,symmetr:[0,12],symmetri:12,synonym:0,synopsi:[3,9],syntact:[0,12],syntax:12,system:9,t1:0,t2:0,t:[0,12],t_co:0,tabl:[0,10,12],table_fea:10,tag:0,take:[0,9,10,12],taken:0,tall:12,tansform:0,target:0,target_col:0,techniqu:10,term:3,terminolog:12,test:12,test_df:12,text:10,textio:0,th:0,than:[0,9,10,12],thank:[0,12],theban:3,thei:[0,12],them:[0,12],themselv:[0,12],then_transform:0,theori:3,therefor:[0,12],thi:[0,3,9,10,12],thing:[0,12],third:10,those:[0,3,9,12],though:12,thought:12,three:12,threshold:[0,12],through:[0,9,12],throughout:12,thu:[3,12],time:[0,3,10,11,12],tip:3,to_csv_kwarg:0,todo:[0,11],togeth:[0,12],toml:0,tool:0,top:[0,11,12],tort:3,total:12,tox:12,trace:[0,8],traceback:12,train:[0,10,11,12],train_df:12,tranform:12,transfer:3,transform:[3,9,10,11],translat:3,transliter:12,treat:0,trick:3,trigger:11,trim:[0,10,12],truli:12,tsvg:12,tupl:0,turn:[9,12],two:[0,12],type:[0,9,10,12],typecheck:0,typeerror:0,typic:12,ubuntu:9,ufunc:12,un:12,unabl:0,unalt:0,under:[0,3,12],underscor:0,understood:0,unfit:[0,12],unicod:0,uniform:10,unintent:0,union:[0,12],uniqu:0,unit:12,univers:[11,12],universalcallchain:0,universalgroup:0,universalpipelin:[0,12],universalpipelineinterfac:0,universaltransform:0,unless:[0,12],unlik:[0,12],unmodifi:12,unnecessari:0,unrel:0,unresolv:0,unresolvedhyperparamet:0,unresolvedhyperparametererror:0,unseen:12,unspecifi:12,unsurprisingli:12,until:[0,12],unweight:[0,12],up:[0,12],upper:[0,10],uppercut:0,uppermost:12,us:[0,3,9],usabl:12,usag:12,user:[0,12],usual:[0,12],util:[0,12],v:0,valid:[0,3,10],valu:[0,12],valueerror:0,vari:0,variabl:[0,12],variou:[0,10,12],ve:12,veri:[0,10,12],version:12,via:[0,12],virtu:12,virtual:9,visual:[0,9],visualizt:0,vs1:10,vs2:10,vs:9,vscode:11,vvs2:10,w_col:[0,12],wa:[0,3,12],wai:[0,3,9,12],want:[0,9,10,12],warn:[0,12],warranti:3,wast:[11,12],we:[0,8,10,11,12],weight:[0,12],well:[0,10,12],were:[0,12],wether:0,what:[0,9,10,12],whatev:[0,12],when:[0,3,12],whenev:[0,12],where:[0,3,11,12],wherebi:12,wherein:[0,12],wherev:[0,12],whether:3,which:[0,3,8,9,10,12],whichev:0,whilst:0,white:0,whole:[0,11,12],whose:[0,10,12],wide:12,wikipedia:12,win_pric:12,winsor:[0,10,12],winsorize_fit:12,winsorize_pric:12,wish:[0,12],with_method:[0,8,12],within:[0,12],without:[0,3,11,12],work:[3,12],workflow:0,workhors:12,worldwid:3,worth:[0,12],would:0,wrap:[0,12],wrapper:12,write:12,write_dataset:0,write_dataset_arg:0,write_pandas_csv:0,writedataset:0,writepandascsv:0,written:[0,12],wrong:0,x64:12,x:[0,10,12],x_col:[0,10,12],xml:12,xxx:[0,5],y:[0,10,12],yet:12,yield:[0,12],you:[0,9,10,11,12],your:[0,3,9,10,12],z:[0,10,12],z_score:[0,10,12],z_score_fit:12,zero:[0,12],zscore:[0,12]},titles:["Frankenfit API reference","Backends and parallel compute","Branching and grouping transforms","Frankenfit Documentation","Cross-validation and hyperparameter search","Working with DataFrames and <code class=\"docutils literal notranslate\"><span class=\"pre\">DataFramePipelines</span></code>","Examples","Hyperparameters","Implementing your own transforms","Installation and getting started","Synopsis and overview","Tips and tricks","Transforms and pipelines"],titleterms:{"2":3,"abstract":12,"break":11,"class":0,"do":11,"import":9,The:[0,12],_appli:8,_fit:8,_submit_appli:8,_submit_fit:8,affix:11,all_col:11,annot:11,api:[0,12],appli:[10,12],assign:11,backend:[0,1,10],base:0,branch:2,bsd:3,call:12,certain:11,chain:12,claus:3,column:11,complex:8,compos:12,comput:[0,1],concaten:12,concis:11,consid:8,consider:8,content:3,convert:11,core:0,creat:10,cross:4,custom:8,dashboard:11,dask:[0,11],daskbackend:11,data:[10,11],datafram:[0,5],dataframepipelin:5,dataset:11,debug:11,declar:8,descript:12,disclaim:3,distribut:10,document:3,exampl:6,fit:[10,11,12],fittransform:12,frankenfit:[0,3,9],futur:0,get:9,go:12,graphviz:9,group:2,hyperparamet:[0,4,7,10,11],if_fit:11,immut:12,implement:8,includ:12,instal:9,larg:11,librari:0,licens:3,local:0,log_messag:11,more:11,name:11,notebook:9,one:12,onli:11,other:12,overview:10,own:8,packag:0,parallel:1,paramet:[8,12],patent:3,piec:11,pipelin:[0,8,10,11,12],prerequisit:9,print:11,re:11,readabl:11,refer:0,run:10,search:4,select:[11,12],simpl:8,start:9,state:11,statefullambda:8,stateless:12,statelesslambda:8,subclass:0,submodul:0,synopsi:10,tabl:3,tag:12,them:10,thing:11,tip:11,togeth:11,trace:11,transform:[0,2,8,12],trick:11,type:11,univers:0,us:[8,10,11,12],usabl:11,valid:4,visual:12,when:11,work:[5,11],write:0,your:[8,11]}})