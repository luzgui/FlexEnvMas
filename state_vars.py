def get_state_vars(self):
     
    
    vars_full= {'tstep':
            {'max':100000,'min':-1.0,'use': True},
    'minutes':
            {'max':1440,'min':-1.0,'use': True},
    'sin':
            {'max':1.0,'min':-1.0,'use': True},
    'cos':
            {'max':1.0,'min':-1.0,'use': True},
    'gen0':# g : PV generation at timeslot
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen1':# g : PV generation forecast next timeslot
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen2':# g : PV generation forecast 1h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen3':# g : PV generation forecast 6h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen4':# g : PV generation forecast 12h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen5':# g : PV generation forecast 24h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen6':# g : PV generation forecast next timeslot
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen7':# g : PV generation forecast 1h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen8':# g : PV generation forecast 6h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen9':# g : PV generation forecast 12h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen10':# g : PV generation forecast 24h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen11':# g : PV generation forecast 12h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'gen12':# g : PV generation forecast 24h ahead
            {'max':self.max_gen,'min':-1.0,'use': True},
    'load0':# g : PV generation at timeslot
            {'max':self.max_load,'min':-1.0,'use': True},
    'load1':# g : PV generation forecast next timeslot
            {'max':self.max_load,'min':-1.0,'use': True},
    'load2':# g : PV loaderation forecast 1h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load3':# g : PV loaderation forecast 6h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load4':# g : PV loaderation forecast 12h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load5':# g : PV loaderation forecast 24h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load6':# g : PV loaderation forecast next timeslot
            {'max':self.max_load,'min':-1.0,'use': True},
    'load7':# g : PV loaderation forecast 1h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load8':# g : PV loaderation forecast 6h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load9':# g : PV loaderation forecast 12h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load10':# g : PV loaderation forecast 24h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load11':# g : PV loaderation forecast 12h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'load12':# g : PV generation forecast 24h ahead
            {'max':self.max_load,'min':-1.0,'use': True},
    'delta0':# g : PV generation at timeslot
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta1':# g : PV generation forecast next timeslot
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta2':# g : PV deltaeration forecast 1h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta3':# g : PV deltaeration forecast 6h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta4':# g : PV deltaeration forecast 12h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta5':# g : PV deltaeration forecast 24h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta6':# g : PV deltaeration forecast next timeslot
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta7':# g : PV deltaeration forecast 1h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta8':# g : PV deltaeration forecast 6h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta9':# g : PV deltaeration forecast 12h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta10':# g : PV deltaeration forecast 24h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta11':# g : PV deltaeration forecast 12h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'delta12':# g : PV generation forecast 24h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess0':# g : PV generation at timeslot
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess1':# g : PV generation forecast next timeslot
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess2':# g : PV excesseration forecast 1h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess3':# g : PV excesseration forecast 6h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess4':# g : PV excesseration forecast 12h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess5':# g : PV excesseration forecast 24h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess6':# g : PV excesseration forecast next timeslot
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess7':# g : PV excesseration forecast 1h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess8':# g : PV excesseration forecast 6h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess9':# g : PV excesseration forecast 12h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess10':# g : PV excesseration forecast 24h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess11':# g : PV excesseration forecast 12h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'excess12':# g : PV generation forecast 24h ahead
            {'max':self.max_load,'min':-self.max_load,'use': True},
    'y': # =1 if ON at t, 0 OTW
        {'max':1.0,'min':-1.0,'use': True},
    'y_1': # =1 if ON in t-1
        {'max':1.0,'min':-1.0,'use': True},
    'y_s':  # +1 if app is schedulled at t (incremental) 
            #(how many times it was connected)
        {'max':self.T,'min':-1.0,'use': True},
    'tar_buy':
        {'max':1.0,'min':-1.0,'use': True},
    'tar_buy0': #Tariff at the next timestep
        {'max':1.0,'min':-1.0,'use': True},
    'E_prof_rem': # reamining energy to supply appliance energy need
        # {'max':2*self.E_prof,'min':-2*self.E_prof}}
        {'max':100.0,'min':-100.0,'use': True}}    
    
        
    vars_list={k:vars_full[k]['use'] for k in vars_full.keys()}       
    #Extract what vars to use features to use    
    vars_to_use={k:vars_full[k] for k in vars_full.keys() if vars_full[k]['use']==True}      
    
    
    #erase 'use' key from dict
    for k in vars_to_use.keys():
        del vars_to_use[k]['use']
    
        
    return vars_to_use, vars_list