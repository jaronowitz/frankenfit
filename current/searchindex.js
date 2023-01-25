Search.setIndex({docnames:["api","backends","branching_and_grouping","cover","crossval","dataframes","examples","hyperparams","implementing_transforms","synopsis","tips_tricks","transforms_and_pipelines"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.md","backends.md","branching_and_grouping.md","cover.md","crossval.md","dataframes.md","examples.md","hyperparams.md","implementing_transforms.md","synopsis.md","tips_tricks.md","transforms_and_pipelines.md"],objects:{"":[[0,0,0,"-","frankenfit"]],"frankenfit.DataFramePipeline":[[0,2,1,"","fit"],[0,3,1,"","fit_transform_class"]],"frankenfit.FitTransform":[[0,2,1,"","apply"],[0,2,1,"","bindings"],[0,2,1,"","materialize_state"],[0,4,1,"","name"],[0,2,1,"","on_backend"],[0,2,1,"","resolved_transform"],[0,2,1,"","state"]],"frankenfit.HP":[[0,2,1,"","resolve"],[0,2,1,"","resolve_maybe"]],"frankenfit.HPCols":[[0,2,1,"","maybe_from_value"],[0,2,1,"","resolve"]],"frankenfit.HPDict":[[0,2,1,"","resolve"]],"frankenfit.HPFmtStr":[[0,2,1,"","resolve"]],"frankenfit.HPLambda":[[0,2,1,"","resolve"]],"frankenfit.Pipeline":[[0,2,1,"","apply"],[0,2,1,"","apply_fit_transform"],[0,2,1,"","if_fitting"],[0,2,1,"","then"],[0,3,1,"","transforms"],[0,2,1,"","with_methods"]],"frankenfit.StatelessTransform":[[0,2,1,"","apply"]],"frankenfit.Transform":[[0,2,1,"","_apply"],[0,2,1,"","_fit"],[0,3,1,"","backend"],[0,2,1,"","fit"],[0,3,1,"","fit_transform_class"],[0,2,1,"","hyperparams"],[0,4,1,"","name"],[0,2,1,"","on_backend"],[0,2,1,"","parallel_backend"],[0,2,1,"","params"],[0,3,1,"","pure"],[0,2,1,"","resolve"],[0,3,1,"","tag"],[0,2,1,"","visualize"]],"frankenfit.UniversalPipeline":[[0,2,1,"","fit"],[0,3,1,"","fit_transform_class"]],"frankenfit.backend":[[0,1,1,"","DaskFuture"]],"frankenfit.core":[[0,1,1,"","ApplyFitTransform"],[0,7,1,"","DEFAULT_VISUALIZE_DIGRAPH_KWARGS"],[0,1,1,"","IfPipelineIsFitting"],[0,1,1,"","LocalFuture"],[0,1,1,"","PipelineMember"]],"frankenfit.core.PipelineMember":[[0,2,1,"","__add__"],[0,2,1,"","_children"],[0,2,1,"","_visualize"],[0,2,1,"","find_by_name"],[0,4,1,"","name"],[0,2,1,"","then"]],"frankenfit.dataframe":[[0,1,1,"","Affix"],[0,1,1,"","Assign"],[0,1,1,"","Clip"],[0,1,1,"","Copy"],[0,1,1,"","Correlation"],[0,1,1,"","DataFrameCallChain"],[0,1,1,"","DataFramePipelineInterface"],[0,1,1,"","DeMean"],[0,1,1,"","Drop"],[0,1,1,"","Filter"],[0,1,1,"","GroupByCols"],[0,1,1,"","ImputeConstant"],[0,1,1,"","ImputeMean"],[0,1,1,"","Join"],[0,1,1,"","Pipe"],[0,1,1,"","Prefix"],[0,1,1,"","ReadDataFrame"],[0,1,1,"","ReadDataset"],[0,1,1,"","ReadPandasCSV"],[0,1,1,"","Rename"],[0,1,1,"","SKLearn"],[0,1,1,"","Select"],[0,1,1,"","Statsmodels"],[0,1,1,"","Suffix"],[0,1,1,"","Winsorize"],[0,1,1,"","WriteDataset"],[0,1,1,"","WritePandasCSV"],[0,1,1,"","ZScore"]],"frankenfit.dataframe.DataFrameCallChain":[[0,2,1,"","affix"],[0,2,1,"","assign"],[0,2,1,"","clip"],[0,2,1,"","copy"],[0,2,1,"","correlation"],[0,2,1,"","de_mean"],[0,2,1,"","drop"],[0,2,1,"","filter"],[0,2,1,"","impute_constant"],[0,2,1,"","impute_mean"],[0,2,1,"","pipe"],[0,2,1,"","prefix"],[0,2,1,"","read_data_frame"],[0,2,1,"","read_dataset"],[0,2,1,"","read_pandas_csv"],[0,2,1,"","rename"],[0,2,1,"","select"],[0,2,1,"","sk_learn"],[0,2,1,"","statsmodels"],[0,2,1,"","suffix"],[0,2,1,"","winsorize"],[0,2,1,"","write_dataset"],[0,2,1,"","write_pandas_csv"],[0,2,1,"","z_score"]],"frankenfit.dataframe.DataFramePipelineInterface":[[0,2,1,"","group_by_cols"],[0,2,1,"","join"]],"frankenfit.universal":[[0,1,1,"","ForBindings"],[0,1,1,"","Identity"],[0,1,1,"","IfFittingDataHasProperty"],[0,1,1,"","IfHyperparamIsTrue"],[0,1,1,"","IfHyperparamLambda"],[0,1,1,"","LogMessage"],[0,1,1,"","Print"],[0,1,1,"","StateOf"],[0,1,1,"","StatefulLambda"],[0,1,1,"","StatelessLambda"],[0,1,1,"","UniversalCallChain"],[0,1,1,"","UniversalPipelineInterface"]],"frankenfit.universal.UniversalCallChain":[[0,2,1,"","identity"],[0,2,1,"","if_fitting_data_has_property"],[0,2,1,"","if_hyperparam_is_true"],[0,2,1,"","if_hyperparam_lambda"],[0,2,1,"","log_message"],[0,2,1,"","print"],[0,2,1,"","stateful_lambda"],[0,2,1,"","stateless_lambda"]],"frankenfit.universal.UniversalPipelineInterface":[[0,2,1,"","for_bindings"],[0,2,1,"","last_state"]],frankenfit:[[0,1,1,"","Backend"],[0,1,1,"","ConstantTransform"],[0,1,1,"","DaskBackend"],[0,1,1,"","DataFramePipeline"],[0,1,1,"","FitTransform"],[0,1,1,"","Future"],[0,1,1,"","HP"],[0,1,1,"","HPCols"],[0,1,1,"","HPDict"],[0,1,1,"","HPFmtStr"],[0,1,1,"","HPLambda"],[0,1,1,"","LocalBackend"],[0,1,1,"","NonInitialConstantTransformWarning"],[0,1,1,"","Pipeline"],[0,1,1,"","StatelessTransform"],[0,1,1,"","Transform"],[0,1,1,"","UniversalPipeline"],[0,5,1,"","UnresolvedHyperparameterError"],[0,6,1,"","columns_field"],[0,6,1,"","dict_field"],[0,6,1,"","fmt_str_field"],[0,6,1,"","params"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","exception","Python exception"],"6":["py","function","Python function"],"7":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:property","5":"py:exception","6":"py:function","7":"py:data"},terms:{"0":[0,9,11],"000000":[9,11],"01":[0,11],"025289":0,"029341":0,"03150901":11,"03466946":11,"035128":11,"0463":11,"05":[0,9,11],"050000":11,"064":11,"07":9,"08":9,"094091":11,"094924":0,"097934":11,"0x7fa2e0a8feb0":11,"0x7fa2e0a92fe0":11,"0x7fa2e0bdbdc0":11,"0x7fa2e0c28970":11,"0x7fa3209eb730":11,"1":[0,9,11],"10":[0,11],"100":[9,11],"1000":9,"100000":11,"101":11,"102":11,"1020":9,"103":11,"104":11,"105":11,"10579":11,"106":11,"106885":11,"107":11,"108":11,"109":11,"11":[9,11],"110":11,"111":11,"112":11,"113":11,"114":11,"11542":9,"119":11,"120":11,"121":11,"122":11,"12226":9,"123":11,"124":11,"125":11,"126":11,"127":11,"1273":11,"130000":11,"1337420":11,"138941":9,"1455":11,"146":11,"147":11,"148":11,"1486":9,"149":11,"150":11,"150929":0,"151":11,"154525":9,"155816":11,"162":11,"163":11,"164":11,"165":11,"166":11,"166911":11,"167":11,"168":11,"169":11,"171":11,"1781":9,"183760":0,"1846":11,"1847":11,"1848":11,"1862":3,"18823":11,"188591":9,"19":9,"190866":11,"194397":9,"195344":0,"199730":11,"2":[0,9,11],"20":[0,9],"200000":11,"2023":3,"2037":9,"204545":11,"206":11,"207":11,"208":11,"209":11,"21":[9,11],"210":11,"211240":11,"212":11,"213":11,"21478":11,"2180":9,"221777":11,"23":[9,11],"23038":11,"230529":9,"2337":9,"235089":11,"2376944":11,"2398":11,"24":11,"243235":11,"249405":11,"25":11,"2537":11,"255136":11,"26970":11,"28":9,"29":[9,11],"2902":9,"3":[9,11],"300786":11,"303313":11,"304961":9,"308130":9,"31":[9,11],"315973":11,"326":[9,11],"327":[9,11],"332419":11,"334":[9,11],"335":[9,11],"336":11,"34":9,"35":9,"3597":11,"3598":11,"3599":11,"36":9,"3600":11,"3605":11,"3606":11,"3608":11,"365591":9,"3794":11,"3795":11,"3802":11,"3803":11,"386357":11,"389973":9,"3932":11,"3934":11,"3982":11,"4":[9,11],"400000":11,"41":9,"412110":9,"4295":9,"43":[9,11],"436":11,"437325":11,"438384":11,"44":11,"443065":9,"448889":11,"457184":11,"45735":11,"457683":11,"459534":11,"464":11,"46553":9,"46890":11,"473374":11,"480529":11,"484358":11,"485492":9,"48794":11,"488533":11,"49907":9,"5":[0,9,11],"50":11,"500000":11,"5028":9,"503841":11,"504049224":11,"50794":11,"5083":11,"508533":11,"5110":9,"5197":9,"522976":9,"5283":11,"5346":11,"536493":9,"539326":9,"542317":11,"542816":11,"5463":11,"55":[9,11],"550595":11,"556029":9,"558533":11,"559616":9,"56":[9,11],"560590":11,"56794":11,"568533":11,"57":11,"574508":11,"5774":11,"5775":11,"5776":11,"5784":11,"58":[9,11],"58794":11,"5883":11,"588533":11,"59":[9,11],"6":[9,11],"600651":11,"6083":11,"61":[9,11],"62":[9,11],"620214":9,"621995":9,"63":[9,11],"6314941":11,"64":[9,11],"640034":11,"6463":11,"65":[9,11],"650595":11,"651208":9,"66":9,"661962":9,"669140":11,"6734":11,"674512":11,"6841":11,"687539":9,"693147":9,"697706":11,"7":[9,11],"70":9,"700000":11,"705":11,"710":11,"711":11,"711384":11,"718":11,"720118":11,"723209":9,"749405":11,"75":[9,11],"753700":11,"757132":9,"76":[9,11],"762":11,"78":11,"785457":9,"788587529099264":11,"789960":11,"79":[9,11],"792993":9,"793014":11,"797287":11,"797940":11,"798533":11,"799722":11,"8":[9,11],"80":11,"800000":11,"814131":11,"817111":11,"82":11,"822812":11,"824":11,"83":11,"836491":11,"84":[9,11],"849405":11,"85":11,"87":11,"877597":11,"89":9,"896522":9,"9":[9,11],"902165627":11,"903136":9,"903625":11,"915621":9,"919174":11,"919813":11,"921591":0,"925196":11,"925839":9,"926416":11,"926595":11,"927241":11,"929112":11,"948906":11,"949405":11,"95":[9,11],"951175":11,"9537":11,"95536379":11,"956":11,"96":11,"968":11,"969":11,"97":11,"971":11,"972":11,"973":11,"973500":9,"974":11,"975":11,"975595":9,"976":11,"977":11,"978":11,"979":11,"98":[9,11],"980":11,"981":11,"99":11,"997759":9,"abstract":0,"break":[0,11],"byte":11,"case":[0,8,9,11],"catch":10,"class":[8,10,11],"default":[0,11],"do":[0,9,11],"final":11,"float":0,"function":[0,10,11],"import":[0,3,8,9,11],"int":0,"long":[3,11],"new":[0,8,11],"null":[0,11],"public":0,"return":[0,8,10,11],"short":[0,9],"static":0,"true":[0,9,11],"try":11,"while":11,A:[0,3,9,11],AND:3,AS:3,And:11,As:[0,8,9,11],At:[0,11],BE:3,BUT:3,BY:3,But:[0,11],By:[0,11],FOR:3,For:[0,3,9,11],IF:3,IN:3,IS:3,If:[0,9,11],In:[0,11],It:[0,9,10,11],Its:0,NO:3,NOT:3,No:[0,11],Not:0,OF:3,ON:3,OR:3,Of:11,One:[0,11],Or:[0,11],SUCH:3,THE:3,TO:3,The:[3,8,9,10],There:11,These:[0,11],To:11,With:[0,9],_:11,__add__:[0,11],__call__:11,__getitem__:0,__init__:11,__name__:11,_appli:0,_base:11,_children:0,_comput:0,_description_:0,_dmn:0,_execute_child:11,_fea:9,_fit:0,_jupyter_mimetyp:11,_pipe_futur:11,_pipe_legaci:11,_pipe_lin:11,_pipe_lines_str:11,_repr_image_svg_xml:11,_repr_mimebundle_:11,_run_input_lin:11,_self:0,_summary_:0,_tool:11,_visual:[0,8],_x:0,_y:0,abbrevi:11,abil:9,abl:[0,11],about:[0,11],abov:[0,3,11],accept:[0,11],access:[0,11],accomplish:11,accord:0,achiev:9,acquir:3,act:[0,11],action:11,actual:[0,10,11],ad:3,add:[0,11],addit:[0,3,11],addition:11,addr:0,address:11,advis:3,affect:0,affix:0,after:[0,11],again:0,against:0,agnost:9,alia:0,all:[0,9,10,11],all_col:[0,11],allow:[0,11],allow_unresolv:0,almost:11,alon:3,along:11,alongsid:11,alpha:11,alread:0,alreadi:[0,3,11],also:[0,9,11],altern:0,although:11,altogeth:11,alwai:[0,11],among:0,an:[0,10,11],analyz:11,ani:[0,3,9,11],anoth:[0,5,9,11],anticip:0,anyth:0,api:[3,9],appear:0,append:[0,11],applend:0,appli:[0,3,10],applic:[0,11],apply_fit_transform:[0,11],apply_fun:[0,11],apply_msg:0,applyfittransform:[0,11],applyresult:0,appropri:0,ar:[0,3,9,10,11],arbitrari:[0,11],arg:[0,11],argument:[0,11],aris:3,arrai:[0,11],art:3,as_index:0,assert:[0,11],assig:0,assign:[0,8,9,11],assist:0,associ:11,assum:0,assumpt:[0,11],astyp:9,attempt:0,attr:0,attrbut:0,attribut:[0,11],aureliu:3,author:[0,8,9],authorship:3,auto:10,automat:[0,11],avail:[0,11],avoid:[0,11],b:3,back:[0,9,11],backend:[3,8,11],bake:3,bake_featur:0,baker:3,balanc:0,bane:3,bar:[0,11],bare:[0,11],base:[9,10,11],base_dir:0,basic:11,batch:9,beauti:3,becaus:[0,11],becom:11,been:[0,8,11],befor:[0,11],begin:11,behavior:[5,11],being:[0,11],belong:0,below:11,benefit:10,bespok:11,best:11,beta:[10,11],between:[0,9,11],beyond:0,bg_fg:0,bias:11,binari:3,bind:[0,9,11],bindings_sequ:0,bit:11,black:0,block:[9,11],bool:0,both:[0,11],bottom:11,bound:0,box:0,bracket:10,branch:[3,9,11],bread:3,breakpoint:10,breviti:[0,9],brief:0,broadli:0,bufsiz:11,build:[9,11],built:[0,10,11],bunch:11,busi:3,c2pread:11,c2pwrite:11,c:[0,8],cach:0,call:[0,9,10],callabl:0,callchain:9,caller:0,camelcas:11,can:[0,9,10,11],cannot:[0,11],capabl:11,captur:0,capture_output:11,carat:[0,9,11],carat_fea:9,care:0,carri:11,categori:11,caus:[0,3,11],certain:[3,11],chain:[0,10],chall:11,chang:0,charg:3,check:[0,10],child:[0,9,11],child_exception_typ:11,children:0,chosen:0,circumst:11,claim:3,clariti:[0,9],class_param:[0,9,11],classmethod:0,clean:[0,11],cleanup:11,clearli:11,client:0,clip:[0,9],close_fd:11,clunki:10,cmd:11,code:[0,3,10],codec:11,coef_:11,coeffici:0,col1:0,col2:0,col:[0,8,10,11],collect:0,color:9,column:[0,8,9,11],columns_field:[0,11],columnstransform:0,com:3,combin:[0,3,9,11],combine_fun:0,combined_model:11,come:11,command:11,common:[0,9,11],compar:0,complet:[0,10,11],complex:[0,9,10,11],complic:11,compos:[9,10],composit:11,compris:11,comput:[3,10,11],concaten:0,conceptu:11,concis:[9,11],concret:0,condit:3,consequenti:3,consid:[0,11],consist:[0,11],constant:[0,11],constantdataframetransform:0,constanttransform:0,constitu:[9,11],construct:[0,11],constructor:[0,11],consult:0,consum:0,contain:[0,11],content:0,context:0,continu:11,contract:[0,3],contrari:3,contribut:3,contributor:3,control:11,conveni:[0,11],convent:[0,9,11],convert:11,copi:[0,10,11],copyright:3,core:[9,11],corr:[0,11],correl:[0,9,11],correland:0,correspond:[0,11],could:[0,9,11],count:11,cours:11,covari:11,cover:11,creat:[0,10,11],creationflag:11,cross:[0,3,9],crucial:11,csv:9,current:0,custom:[0,11],cut:[0,9],cwd:11,d:[0,9,11],damag:3,dask_futur:0,daskbackend:0,daskfutur:0,data:[0,3,5,11],data_appli:[0,8],data_fit:[0,8],datafram:[3,8,9,11],dataframecallchain:0,dataframegroup:0,dataframepipelin:[0,3,9,10,11],dataframepipelineinterfac:0,dataframetransform:0,dataset:[0,9,11],dataset_kwarg:0,datatyp:0,de:[0,8,11],de_mean:[0,11],declar:0,decor:11,def:[0,8,11],default_visualize_digraph_kwarg:0,defer:[0,11],defin:[0,9,11],degrad:11,delete_match:0,demean:[0,8,11],depend:0,deprec:11,deprecate_positional_arg:11,depth:[0,9,11],depth_fea:9,deriv:11,describ:[9,11],descript:9,design:0,desir:[0,3],dest:0,dest_col:0,destin:0,detail:11,determin:[0,11],deviat:[0,11],devis:11,df:[0,9,11],df_appli:0,df_fit:0,df_in:9,df_oo:11,df_out:9,diamet:11,diamond:[0,9,11],diamond_model:9,diamonds_df:[0,11],dict:[0,9],dict_field:0,dictcomp:11,dictionari:0,did:11,differ:[0,9,11],difficult:11,digraph:[0,11],digraph_kwarg:0,direct:[3,11],directli:[0,11],directori:11,discard:0,discuss:11,disjoint:11,displai:11,distinct:[0,11],distort:11,distribut:[0,3,11],divid:[9,11],dmn:11,dmn_2col:11,dmn_price_weight:11,doc:11,docstr:0,document:[0,11],doe:[0,11],doesn:11,dollar:9,domain:[0,9,11],don:[0,11],dot:11,dot_command:11,doubt:11,doubtlessli:11,drop:[0,11],dtype:11,duplic:11,e:[0,9,11],each:[0,3,9,11],eampl:10,earlier:11,easi:9,easili:11,eat:3,edge_attr:0,effect:[0,11],effici:[0,11],either:0,element:0,els:[0,11],emb:11,embed:[0,11],emit:0,emoji:0,empti:[0,5,11],empty_pipelin:11,enabl:[0,11],encapsul:[9,11],encod:11,end:11,engin:11,enhanc:10,enoent:11,ensur:[0,11],entir:[0,9,11],env:11,environ:0,equal:[0,11],equival:[0,11],err_filenam:11,err_msg:11,errno:11,errno_num:11,error:[10,11],errread:11,errwrit:11,especi:11,essenti:11,estim:[9,11],estoppel:3,etc:0,eval_df:11,eval_oos_df:11,evalu:[10,11],even:[0,3,10,11],event:3,everi:[0,11],exact:11,exactli:[0,11],exampl:[0,3,9,10,11],except:[0,3,11],excit:3,exclud:11,exclus:3,execut:[0,9,11],executablenotfound:11,exemplari:3,exist:11,existing_data_behavior:0,exp_price_hat:11,exp_price_hat_fit:11,expect:[0,11],explicit:[0,11],explicitli:0,exploit:9,expm1:[9,11],exponenti:11,expos:0,express:[0,3,10],expressli:3,extend:0,extens:11,extra_group:11,extract:10,extraordinari:11,extrem:11,f:[9,11],facet:11,fact:[0,11],factori:11,fail:[0,11],failur:3,fair:9,fals:0,famili:11,far:11,fashion:3,featur:[9,11],feed:[9,11],few:11,fewer:0,ff:[0,8,9,10,11],field:[0,11],figsiz:[9,11],figur:11,file:[0,5,11],filenotfounderror:11,filepath:0,filter:[0,11],filter_fun:0,find:0,find_by_nam:[0,11],find_by_tag:0,first:[0,11],fit:[0,3],fit_diamond_model:9,fit_dmn:11,fit_fun:0,fit_group_on_all_other_group:9,fit_group_on_self:0,fit_intercept:[0,9,11],fit_model:11,fit_msg:0,fit_regress:11,fit_transform:0,fit_transform_class:0,fitdataframetransform:[0,11],fitting_schedul:[0,9],fittransform:[0,9],fituniversaltransform:0,fitzscor:0,fix:11,flag:0,float64:11,fmt_str_field:0,fo:0,focu:9,fold:9,follow:[0,3,11],fontnam:0,fontsiz:0,foo:[0,11],foobar:0,for_bind:0,forbind:0,form:3,formal:11,format:[0,11],formatt:11,former:11,found:0,four:11,frac:11,frame:[0,5],frankefit:11,frankenfit:[8,9,10,11],free:[0,3,10],freeli:0,frequenc:0,from:[0,3,9,10,11],full:0,fulli:0,fun:0,func:11,fundament:9,furthermor:[0,11],g:[0,11],g_co:0,gener:[0,9,10,11],general:0,georg:3,get:[0,11],get_real_method:11,getattr:11,getlogg:0,gid:11,github:3,give:[9,11],given:[0,11],good:[3,9,11],grant:3,graph:11,graphviz:[0,11],greater:[0,11],ground:11,group:[0,3,9,11],group_by_col:[0,9],group_kei:[0,9],groupbycol:0,grouper:0,h:9,ha:[0,11],half:11,hand:[0,11],handi:11,handl:11,happen:11,hat_col:[0,9,11],have:[0,3,8,11],head:[9,11],heavyweight:11,height:0,henc:11,here:[0,9,11],hereaft:3,herebi:3,herself:0,heterogen:[0,11],high:[9,11],him:11,hist:[9,11],hold:[0,11],holder:3,home:9,homepag:3,hood:11,hostedtoolcach:11,how:[0,11],howev:[0,3,11],hp:0,hp_name:0,hpcol:[0,10],hpdict:0,hpfmtstr:[0,11],hplambda:0,http:3,hyerparam:8,hypeparamet:0,hyperparam:[0,10],hyperparamet:[3,11],i:[0,9],id:10,ideal:9,ident:[0,10],identifi:[0,11],idiomat:[0,11],if_fit:[0,11],if_fitting_data_has_properti:0,if_hyperparam_is_tru:0,if_hyperparam_lambda:0,iffittingdatahasproperti:0,ifhyperparamistru:0,ifhyperparamlambda:0,ifpipelineisfit:0,ignor:0,illustr:11,imag:9,imagin:11,immedi:11,implement:[0,3,9,11],implementatino:0,impli:3,implic:3,implicit:11,impute_const:[0,9],impute_mean:0,imputeconst:[0,11],imputemean:0,incident:[3,11],includ:[0,3,9],inconveni:11,incorpor:11,increment:11,indent:0,independ:[0,11],index:[0,8,9,10,11],index_col:0,index_label:0,indic:[0,11],indirect:3,individu:11,ineffici:11,influenc:0,info:0,infring:3,ing:0,inher:9,inherit:11,initi:[0,9,10,11],inner:0,input:[0,9,11],input_encod:11,input_lin:11,inspect:11,inspir:[9,11],instanc:[0,3,11],instanti:[0,11],instead:[0,11],int32:9,intend:9,intent:0,interact:0,intercept:0,intercept_:11,interepret:0,interest:11,intern:11,interpret:10,interrupt:3,introduc:11,invent:11,invok:[0,11],involv:11,io:11,ipython:11,irrevoc:3,item:11,iter:0,its:[0,9,11],itself:[0,9],j:9,job:11,join:[0,11],jupyter_integr:11,jupyterintegr:11,just:[0,11],keep:0,keep_larg:11,keepcolumn:0,kei:0,kendal:0,kept:11,keyerror:0,keyword:[0,11],kind:[0,10,11],know:11,known:[0,10,11],kwarg:[0,11],labori:11,lambda:[0,9,11],languag:[0,9,11],larger:11,largest:11,last:[0,11],last_stat:0,later:11,latter:11,layer:[0,11],layout:11,lead:0,learn:[0,9,11],least:[0,11],left:0,left_col:0,left_on:0,len:[0,9],length:[0,11],less:0,let:11,level:[0,10],liabil:3,liabl:3,lib:11,librari:[9,10,11],light:11,lightweight:9,like:[0,9,10,11],likewis:0,limit:[0,3,11],linear:[9,11],linear_model:[9,11],linearregress:[0,9,11],link:0,list:[0,3,8,11],liter:0,littl:[0,11],ll:11,load:[10,11],loc:11,local:11,localbackend:0,localfutur:0,log1p:[0,9,11],log:[0,9,11],log_messag:0,log_pric:11,log_price_cara_fit:11,log_price_carat:11,log_price_carat_fit:11,log_price_fit:11,log_price_lambda:11,logger:0,logger_nam:0,logic:[0,10,11],logmessag:0,look:11,lookup:11,loss:3,low:9,lower:[0,9],made:[0,3],magic:0,mai:[0,9,11],main:[0,10,11],make:[0,3,10,11],manag:0,mani:[0,9,11],manipul:11,manner:[3,11],manual:11,map:0,marcu:3,mark:0,match:11,materi:3,materialize_st:0,max:[0,3,11],maxban:3,maybe_from_valu:0,mean:[0,8,11],meant:[0,11],medit:3,memori:10,mention:11,merchant:3,mere:11,messag:0,met:3,method:[0,8,10,11],method_nam:11,meticul:10,might:11,mime_typ:11,mimebundleformatt:11,mimetyp:11,min:[0,11],min_ob:0,mind:0,minimum:0,miniscul:11,miss:[0,11],mix:11,mode:0,model:[0,9,10,11],modif:3,modifi:11,modul:[0,11],moment:11,monospac:0,more:[0,9,11],most:[0,8,11],much:11,multi:0,multipl:11,must:[0,3,8,11],mutat:0,my_model:0,my_pipelin:[0,11],mycustomtransform:0,mygreattransform:11,mypi:[0,10],naiv:11,name:[0,8,9,11],nameerror:0,namespac:[0,11],nan:0,ndarrai:0,neato_no_op:11,necessari:0,necessarili:3,need:[0,11],neglig:3,nest:11,never:[0,11],next:[0,11],ngroup:9,no_cach:0,node_attr:0,non:[0,3,11],nonc:0,none:[0,11],noninitialconstanttransformwarn:0,nor:11,notat:11,note:[0,11],noth:[0,11],notic:[3,11],notimplementederror:0,now:[0,11],np:[0,9,11],nuanc:11,number:[0,11],numer:0,numpi:[0,9,11],obj:[0,11],object:[0,9,11],oblig:10,observ:[0,9,11],obtain:[0,9],occur:0,offer:3,often:[0,11],old:0,omit:[0,11],on_backend:0,onc:[0,9,11],one:[0,9],ones:0,onli:[0,3,9,11],onto:[0,11],oo:11,op:0,open:[0,3],oper:[0,9,11],opt:11,option:[0,11],order:[0,11],ordinari:11,ordinarili:[0,11],orer:0,org:3,organ:11,origin:11,os:11,oserror:11,other:[0,3,9],other_pipelin:0,otherwis:[0,3,11],ought:0,our:11,out:[0,3,9,11],outer:0,outlier:[0,9,11],output:[0,9,11],outsid:[0,11],over:11,overrid:[0,8],overridden:11,overview:[0,3],overwrit:0,own:[0,3,10,11],p1:11,p2:11,p2cread:11,p2cwrite:11,p:0,p_co:0,packag:11,pair:0,panda:[0,8,9,11],parallel:[3,8,9],parallel_backend:0,param:[0,8,10,11],paramet:0,parameter:0,parent:0,parquet:0,part:[0,3,11],parti:9,particular:[0,3,11],particularli:[0,11],partitioning_schema:0,pass:[0,11],pass_fd:11,path:[0,11],pavilion:11,pd:[0,8],pearson:0,peculiar:3,per:0,percent:0,percentil:0,perform:[9,11],perhap:11,permit:3,perpetu:3,perspect:0,pick:11,pip:[0,11],pipe:[0,9,11],pipe_lines_str:11,pipelin:[3,5],pipelinegroup:0,pipelinememb:0,pipes:11,piplin:11,plan:11,pleas:0,plot:[9,11],plugin:0,point:0,pollut:0,popen:11,posit:11,position:11,posixpath:11,possibl:[0,3,9,11],potenti:[0,11],power:11,practic:11,preced:[0,9],precis:0,predict:[0,9,11],predict_pric:11,predict_price_tag:11,predictor:[0,11],preexec_fn:11,prefer:[0,11],prefix:0,premium:9,prepar:[0,9,11],prepare_featur:11,prepare_training_respons:11,prepend:0,presenc:11,preserv:11,preservs:11,pretti:11,prevent:11,previou:11,previous:11,price:[0,9,11],price_dmn2:0,price_hat:[0,9,11],price_hat_dollar:[9,11],price_model:11,price_model_corr:11,price_model_corr_fit:11,price_orig:[0,11],price_rank:0,price_regress:11,price_scal:0,price_train:[0,9,11],price_win2:0,price_win:0,principl:11,print:0,print_method:11,problem:11,proc:11,proce:11,procedur:11,process:11,procur:3,produc:[0,11],product:[0,10],profit:3,project:[0,3],prone:11,properti:0,provid:[0,3,8,9,11],pull:[0,11],pure:0,purpos:[3,11],put:11,py:11,pyarrow:[0,10],pydataset:[0,9,11],pyproject:0,python3:11,python:[9,11],quantil:11,queri:11,question:[9,11],quiet:11,quit:11,r:[0,11],r_co:0,rais:[0,11],random:[0,9,11],random_st:11,randomli:9,rang:[10,11],rare:0,rather:[0,11],raw:11,re:[0,11],read:[0,5,11],read_csv_arg:0,read_data_fram:0,read_dataset:[0,10],read_pandas_csv:[0,9],readabl:[9,11],readdatafram:0,readdataset:0,reader:[0,10,11],readi:11,readpandascsv:0,real:11,reason:[0,11],recal:11,receiv:[3,11],recent:11,recommend:[0,9],recomput:0,recurs:0,redistribut:3,reduc:11,refer:[3,11],referenc:0,regardless:11,regress:[0,9,10,11],regress_fit:11,relat:0,reli:0,rememb:11,remov:0,renam:0,render:[0,11],replac:[0,11],repo:9,repr:11,repres:[0,11],reproduc:3,requir:[0,11],resampl:9,research:0,reset_index:9,resolut:[0,11],resolv:0,resolve_fun:0,resolve_mayb:0,resolved_transform:[0,11],respect:[0,8,9,11],respons:[0,9,10,11],response_col:[0,9,11],restore_sign:11,restrict:0,result:[0,9,10,11],retain:3,retriev:11,reusabl:9,revisit:11,rewritten:11,right:[0,3],right_col:0,right_on:0,rogu:11,row:[0,9,11],royalti:3,run:0,run_check:11,runner:9,runtim:11,runtimewarn:0,s:[0,3,9,10,11],safe:0,sai:[0,11],sake:[0,9],same:[0,11],sampl:[0,9,11],sample_weight:0,satisfi:3,saw:11,scalar:0,scale:11,scanner_kwarg:0,scatter:[9,11],schedul:0,schema:11,scikit:[0,9,11],score:[0,9,10,11],score_corr:11,score_ms:11,score_predict:11,scratch:0,screen:0,screenshot:10,se:0,search:[3,9,11],section:11,see:[0,11],seem:11,select:0,self:[0,8,11],selfdpi:0,selfupi:0,sell:3,send:11,sens:[10,11],separ:[0,11],sequenc:[0,9,11],sequenti:[0,11],seri:[0,8,9,11],serv:[0,11],servic:3,set:[0,9,11],setup:11,sever:11,shall:3,shallow:0,shape:[0,11],shell:11,should:[0,10,11],show:11,si1:9,si2:9,side:[0,11],signatur:[0,11],similar:11,similarli:0,simpl:[0,11],simplest:11,simpli:0,simplifi:0,sinc:[0,11],singl:[0,11],site:11,situat:[0,11],size:[9,11],sk_learn:[0,9,11],skip:11,sklearn:[0,9,11],sklearn_class:[0,11],sleek:10,slice:0,sm_class:0,small:0,smaller:11,snake_cas:11,so:[0,10,11],softwar:3,some:[0,3,8,9,10,11],someth:11,somewher:[0,11],sooner:10,sort:0,sourc:[0,3,11],speak:[0,11],spearman:0,special:[3,11],specif:[0,9,11],specifi:[0,11],split:3,sqrt:[0,11],squar:11,stacklevel:11,standard:[0,9,11],star:0,start:[0,11],start_new_sess:11,startupinfo:11,state:[0,3,8,9,11],stateful_lambda:0,statefullambda:0,stateless:0,stateless_lambda:[0,11],statelessdataframetransform:0,statelesslambda:[0,11],statelesstransform:[0,11],statement:0,stateof:0,statist:9,statlesstransform:11,statsmodel:[0,9],std:11,stderr:11,stdin:11,stdin_writ:11,stdout:[0,11],step:11,still:[0,11],store:[0,11],str:[0,8,10,11],strerror:11,strict:3,strictli:11,strike:0,string:[0,11],structur:11,style:[0,9,11],stylist:[0,9],sub:[0,10,11],subclass:11,subclass_nam:0,subject:3,submodul:11,subprocess:11,subsampl:11,subset:[0,11],substitut:3,subtract:11,success:11,succinctli:11,suffix:[0,9,10],sugar:[0,11],suggest:11,summar:11,summari:10,suppli:[0,11],support:11,supported_numb:11,suppos:[0,9,11],sure:11,surfac:3,svg:11,svg_encod:11,symbol:0,symmetr:[0,11],symmetri:11,synonym:0,synopsi:3,syntact:[0,11],syntax:11,system:11,t1:0,t2:0,t:[0,11],t_co:0,tabl:[0,9,11],table_fea:9,tag:0,take:[0,9,11],taken:0,tall:11,tansform:0,target:0,target_col:0,techniqu:9,term:3,terminolog:11,test:11,test_df:11,text:[9,11],textio:0,textiowrapp:11,th:0,than:[0,9,11],thank:[0,11],theban:3,thei:[0,11],them:[0,11],themselv:[0,11],then_transform:0,theori:3,therefor:[0,11],thi:[0,3,9,11],thing:[0,11],third:9,those:[0,3,11],though:11,thought:11,three:11,threshold:[0,11],through:[0,11],throughout:11,thu:[3,11],time:[0,3,9,10,11],tip:3,to_csv_kwarg:0,todo:[0,10],togeth:[0,11],toml:0,tool:0,top:[0,10,11],tort:3,total:11,tox:11,trace:[0,8],traceback:11,train:[0,9,10,11],train_df:11,tranform:11,transfer:3,transform:[3,9,10],translat:3,transliter:11,treat:0,trick:3,trigger:10,trim:[0,9,11],truli:11,tupl:0,turn:11,two:[0,11],type:[0,9,11],typecheck:0,typeerror:0,typic:11,ufunc:11,uid:11,umask:11,un:11,unabl:0,unalt:0,under:[0,3,11],underscor:0,understood:0,unfit:[0,11],unicod:0,uniform:9,unintent:0,union:[0,11],uniqu:0,unit:11,univers:[10,11],universal_newlin:11,universalcallchain:0,universalgroup:0,universalpipelin:[0,11],universalpipelineinterfac:0,universaltransform:0,unless:[0,11],unlik:[0,11],unmodifi:11,unnecessari:0,unrel:0,unresolv:0,unresolvedhyperparamet:0,unresolvedhyperparametererror:0,unseen:11,unspecifi:11,unsurprisingli:11,until:[0,11],unweight:[0,11],up:[0,11],upper:[0,9],uppercut:0,uppermost:11,us:[0,3],usabl:11,usag:11,user:[0,11],usual:[0,11],util:[0,11],v:0,valid:[0,3,9],valu:[0,11],valueerror:0,vari:0,variabl:[0,11],variou:[0,9,11],ve:11,veri:[0,9,11],version:11,via:[0,11],virtu:11,visual:0,visualizt:0,vs1:9,vs2:9,vscode:10,w_col:[0,11],wa:[0,3,11],wai:[0,3,11],want:[0,9,11],warn:[0,11],warranti:3,wast:[10,11],we:[0,8,9,10,11],weight:[0,11],well:[0,9,11],were:[0,11],wether:0,what:[0,9,11],whatev:[0,11],when:[0,3,11],whenev:[0,11],where:[0,3,10,11],wherebi:11,wherein:[0,11],wherev:[0,11],whether:3,which:[0,3,8,9,11],whichev:0,whilst:0,white:0,whole:[0,10,11],whose:[0,9,11],wide:11,wikipedia:11,win_pric:11,winsor:[0,9,11],winsorize_fit:11,winsorize_pric:11,wish:[0,11],with_method:[0,8,11],within:[0,11],without:[0,3,10,11],work:[3,11],workflow:0,workhors:11,worldwid:3,worth:[0,11],would:0,wrap:[0,11],wrapper:11,write:11,write_dataset:0,write_dataset_arg:0,write_pandas_csv:0,writedataset:0,writepandascsv:0,written:[0,11],wrong:0,x64:11,x:[0,9,11],x_col:[0,9,11],xml:11,xxx:[0,5],y:[0,9,11],yet:11,yield:[0,11],you:[0,9,10,11],your:[0,3,9,11],z:[0,9,11],z_score:[0,9,11],z_score_fit:11,zero:[0,11],zscore:[0,11]},titles:["Frankenfit API reference","Backends and parallel compute","Branching and grouping transforms","Frankenfit Documentation","Cross-validation and hyperparameter search","Working with DataFrames and <code class=\"docutils literal notranslate\"><span class=\"pre\">DataFramePipelines</span></code>","Examples","Hyperparameters","Implementing your own transforms","Synopsis and overview","Tips and tricks","Transforms and pipelines"],titleterms:{"2":3,"abstract":11,"break":10,"class":0,"do":10,The:[0,11],_appli:8,_fit:8,_submit_appli:8,_submit_fit:8,affix:10,all_col:10,annot:10,api:[0,11],appli:[9,11],assign:10,backend:[0,1,9],base:0,branch:2,bsd:3,call:11,certain:10,chain:11,claus:3,column:10,complex:8,compos:11,comput:[0,1],concaten:11,concis:10,consider:8,content:3,convert:10,core:0,creat:9,cross:4,custom:8,dashboard:10,dask:[0,10],daskbackend:10,data:[9,10],datafram:[0,5],dataframepipelin:5,dataset:10,debug:10,declar:8,descript:11,disclaim:3,distribut:9,document:3,exampl:6,fit:[9,10,11],fittransform:11,frankenfit:[0,3],futur:0,go:11,group:2,hyperparamet:[0,4,7,9,10],if_fit:10,immut:11,implement:8,includ:11,larg:10,librari:0,licens:3,local:0,log_messag:10,more:10,name:10,one:11,onli:10,other:11,overview:9,own:8,packag:0,parallel:1,paramet:[8,11],patent:3,piec:10,pipelin:[0,8,9,10,11],print:10,re:10,readabl:10,refer:0,run:9,search:4,select:[10,11],simpl:8,state:10,stateless:11,subclass:0,submodul:0,synopsi:9,tabl:3,tag:11,them:9,thing:10,tip:10,togeth:10,trace:10,transform:[0,2,8,11],trick:10,type:10,univers:0,us:[8,9,10,11],usabl:10,valid:4,visual:11,when:10,work:[5,10],write:0,your:[8,10]}})