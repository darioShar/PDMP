    def forward(self, data, t, speed = None):
        #new_data = data.clone()
        #time_horizons = t.clone().detach()

        # for the moment, everything is happening on the cpu
        if speed is None:
            if self.sampler == 'ZigZag':
                speed = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*data.shape[0],))]
                speed = speed.reshape(data.shape[0], 1, 2)
            elif self.sampler == 'HMC':
                speed = torch.randn_like(data).to(self.device)
            elif self.sampler == 'BPS':
                speed = torch.randn(data.shape)
        
        # can put everyhting on the device
        speed = speed.to(self.device)

        if self.sampler == 'ZigZag':
            while (t > 0.).any():
                self.ZigZag_gauss_1event(data, speed, t)
        elif self.sampler == 'HMC':
            while (t > 0.).any():
                self.HMC_gauss_1event(data, speed, t)
                #speed = torch.randn_like(data)
                speed[t > 0] = torch.randn_like(speed[t > 0])
        elif self.sampler == 'BPS':
            while (t > 0.).any():
                self.BPS_gauss_1event(data, speed, t)

        return data, speed


    def training_losses_zigzag(self, model, X_t, V_t, t):
        # send to device
        X_t = X_t.to(self.device)
        V_t = V_t.to(self.device)
        t = t.to(self.device)
        
        # tensor to give as input to the model. It is the concatenation of the position and the speed.
        X_V_t = torch.concat((X_t, V_t), dim = -1)
        #print(time_horizons[0], X_V_t[0])
        #print(X_V_t.mean(dim = 0), X_V_t.std(dim = 0))

        # run the model
        output = model(X_V_t, t)

        assert X_t.shape[-1] == 2, 'only dimension 2 is implemened'
        # invert time on component 1 and 2
        X_V_inv_t_0 = X_V_t.detach().clone() # clone to avoid modifying the original tensor, detach to avoid computing gradients on original tensor
        X_V_inv_t_1 = X_V_t.detach().clone()
        X_V_inv_t_0[:, :, 2] *= -1 # reverse speed on i = 1
        X_V_inv_t_1[:, :, 3] *= -1 # reverse speed on i = 2

        # run the model on each inverted speed component
        output_inv_0 = model(X_V_inv_t_0, t)
        output_inv_1 = model(X_V_inv_t_1, t)
        
        # compute the loss
        def g(x):
            return (1 / (1+x))
        loss = g(output[:, :, 0])**2 + g(output_inv_0[:, :, 0])**2
        loss += g(output[:, :, 1])**2 + g(output_inv_1[:, :, 1])**2
        loss -= 2*(g(output[:, :, 0]) + g(output[:, :, 1]))
        return loss



    def training_losses(self, model, X_batch, V_batch = None, time_horizons = None):
                # generate random speed
        #if self.sampler == 'ZigZag':
        #    Vbatch = torch.tensor([-1., 1.])[torch.randint(0, 2, (2*X_t.shape[0],))]
        #    Vbatch = Vbatch.reshape(X_t.shape[0], 1, 2)
        #elif self.sampler == 'HMC':
        #    Vbatch = torch.randn_like(X_t)
        #elif self.sampler == 'BPS':
        #    Vbatch = torch.randn_like(X_t)
        
        # generate random time horizons
        if time_horizons is None:
            time_horizons = self.T * (torch.rand(X_batch.shape[0])**2)

        # must be of the same shape as Xbatch for the pdmp forward process, since it will be applied component wise
        t = time_horizons.clone().detach().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
        
        # clone our initial data, since it will be modified by forward process
        x = X_batch.clone()
        #v = Vbatch.clone()
        
        # actually faster to switch to cpu for forward process in the case of pdmps
        device = self.device
        self.device = 'cpu'
        # put everyhting on the device
        X_batch = X_batch.to(self.device)
        t = t.to(self.device)
        #print('t', t[0])

        # apply the forward process. Everything runs on the cpu.
        X_t, V_t = self.forward(X_batch, t, speed = V_batch)
        
        #print('x', x[0])
        #print('X_t', X_t[0])
        # put back on the device
        self.device = device

        # check that the data has been modified
        #idx = (x == Xbatch)
        #print(idx, x[idx], Xbatch[idx])
        #idx = (v == Vbatch)
        #print(idx, v[idx], Vbatch[idx])
        #assert ((x == X_t).logical_not()).any() and ((v == Vbatch).logical_not()).any()
        #assert not ((x == X_t)).any()
        # check that the time horizon has been reached for all data
        assert not (t != 0.).any()

        if self.sampler == 'ZigZag':
            losses = self.training_losses_zigzag(model, X_t, V_t, time_horizons)
        elif self.sampler == 'HMC':
            losses = self.training_loss_hmc(model, X_t, V_t, time_horizons)
        elif self.sampler =='BPS':
            losses = self.training_loss_bps(model, X_t, V_t, time_horizons)
        
        return losses.mean()

    def splitting_zzs_DBD(self, 
                          model, 
                          T, 
                          N, 
                          shape = None, 
                          x_init=None, 
                          v_init=None, 
                          print_progession = False, 
                          get_sample_history = False):
        print('ZigZag generation')
        print(get_sample_history)
        if print_progession:
            print_progession = lambda x : tqdm(x)
        else:
            print_progession = lambda x : x
    
        timesteps = torch.linspace(1, 0, N+1)**2 * T
        #timesteps = timesteps.flip(dims = (0,))
        #times = T - deltas.cumsum(dim = 0)
        assert (shape is not None) or (x_init is not None) or (v_init is not None) 
        if x_init is None:
            x_init = torch.randn(*shape)
        if v_init is None:
            v_init = torch.tensor([-1., 1.])[torch.randint(0, 2, (x_init.shape[0],))].reshape(x_init.shape[0], *[1]*len(x_init.shape[1:])).repeat(1, *(x_init.shape[1:]))
        #chain = [pdmp.Skeleton(torch.clone(x_init), torch.clone(v_init), 0.)]
        chain = []
        x = x_init.clone()
        v = v_init.clone()
        model.eval()
        with torch.inference_mode():
            for i in print_progession(range(int(N))):
                time = timesteps[i]
                delta = (timesteps[i] - timesteps[i+1]) if i < N - 1 else timesteps[i]
                if get_sample_history:
                    chain.append(torch.concat((x, v), dim = -1))
                # compute x_n-1 from x_n
                x -= v * delta / 2 # x - v * δ / 2
                time_mid = time - delta/ 2 #float(n * δ - δ / 2) #float(n - δ / 2)
                density_ratio = model(torch.concat((x,v), dim = -1).to(self.device),
                                    (torch.ones(x.shape[0])*time_mid).to(self.device))[..., :2]
                #print(density_ratio.mean(dim=0))
                switch_rate = density_ratio.cpu()* torch.maximum(torch.zeros(x.shape), -v * x)
                self.flip_given_rate(v, switch_rate, delta)
                x -= v * delta / 2 #x - v * δ / 2
                #print(x, v)
                #chain.append(Skeleton(x.copy(), v.copy(), n * δ))
        if get_sample_history:
            chain.append(torch.concat((x, v), dim = -1))
            print(chain[-1].shape)
            return torch.stack(chain)
        return torch.concat((x, v), dim = -1)
    
    def refresh_given_rate(self, x, v, t, lambdas, s, model):
        # lambdas[lambdas == 0.] += 1e-9
        event_time = torch.distributions.exponential.Exponential(lambdas)
        temp = event_time.sample()
        #temp = temp.reshape(-1, *([1]*len(x.shape[1:]))).repeat(1, *x.shape[1:])
        tmp = model(torch.cat(
            (x,
             t * torch.ones(x.shape[0], 1, 1).to(self.device)),
               dim = -1).to(self.device)
            ).sample()
        # print(temp[temp <= s].shape)
        # print((tmp[temp <= s]))
        v[temp <= s] = tmp[temp <= s]
        
    def reverse_sampling(self,
                        model,
                        reverse_steps=None,
                        shape = None,
                        nsamples = None,
                        time_spacing = None,
                        initial_data = None, # sample from Gaussian, else use this initial_data
                        print_progression = False,
                        print_progession = False,
                        get_sample_history = True, #### SET TO FALSE
                        ):
        assert initial_data is None, 'Using specified initial data is not yet implemented.'
        assert nsamples is not None or shape is not None
        if nsamples is not None:
            print('ATTENTION: pdmp reverse_samlping, lin 330, only dim 2')
            shape = (nsamples, 1, 2)
        
        reverse_sample_func = {
            'ZigZag': self.splitting_zzs_DBD,
            'HMC': self.splitting_HMC_DRD,
            'BPS': self.splitting_BPS_RDBDR
        }
        # for the moment, don;t do anything with time spacing
        assert time_spacing is None, 'Specific time spacing is not yet implemented.'

        samples_or_chain = reverse_sample_func[self.sampler](model,
                                              self.T, 
                                              reverse_steps if reverse_steps is not None else 100, # for ZigZAG REMOVE,
                                              shape = shape,
                                              print_progession=print_progession,
                                              get_sample_history = get_sample_history)
        return samples_or_chain

