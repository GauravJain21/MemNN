-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('cunn')
require('nngraph')
paths.dofile('LinearNB.lua')

--input: params ()

local function build_memory(params, input, context, time)
    local hid = {}
    hid[0] = input
    local shareList = {}
    shareList[1] = {}
    --convolutional layer with width of convolution = 1, for each index
    --it outputs a tensor of size params.edim. Now, these are concatenated
    --to form a tensor of size nwords * edim. 
    --utility: transforms the vocab indices into tensors via convolution
    --opeartions. In short, each word among nwords get a representation
    --in the edim dimension plane. (d X V embedding matrix according to
    --paper)
    local Ain_c = nn.LookupTable(params.nwords, params.edim)(context)
    local Ain_t = nn.LookupTable(params.memsize, params.edim)(time)
    local Ain = nn.CAddTable()({Ain_c, Ain_t})
    --B = C
    local Bin_c = nn.LookupTable(params.nwords, params.edim)(context)
    local Bin_t = nn.LookupTable(params.memsize, params.edim)(time)
    local Bin = nn.CAddTable()({Bin_c, Bin_t})
    --MM is for matrix multiplication, referred to as dot product in
    --the paper. false means take the transpose of first argument.
    --cuda is for performing this operation on GPU.
    for h = 1, params.nhop do
        --View reshapes the object to achieve minibatch training
        --total dimensions = 1, 1 repsents batched mode, -1 
        local hid3dim = nn.View(1, -1):setNumInputDims(1)(hid[h-1])
        local MMaout = nn.MM(false, true):cuda()
        --dot product of last layer's output (hid3dim(u)) with Ain (same
        -- for all layers (using layer wise (RNN like) appoach)).
        local Aout = MMaout({hid3dim, Ain})
        local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)   --number of dimensions are 2
        local P = nn.SoftMax()(Aout2dim)                        --apply softmax to obtain probabilities
        -- pi = Softmax(u^T.mi)
        local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)  --reshape probability tensor
        local MMbout = nn.MM(false, false):cuda()
        --Bout = o
        local Bout = MMbout({probs3dim, Bin})                   --multiply embedding B with probability
        --C = u
        local C = nn.LinearNB(params.edim, params.edim)(hid[h-1])
        table.insert(shareList[1], C)
        --Output of this layer = u + o = C + Bout
        local D = nn.CAddTable()({C, Bout})
        if params.lindim == params.edim then                    --no activation applied
            hid[h] = D
        elseif params.lindim == 0 then
            hid[h] = nn.ReLU()(D)                               --ReLU applied to half the samples
        else
            local F = nn.Narrow(2,1,params.lindim)(D)
            local G = nn.Narrow(2,1+params.lindim,params.edim-params.lindim)(D)
            local K = nn.ReLU()(G)
            hid[h] = nn.JoinTable(2)({F,K})
        end
    end

    return hid, shareList
end

function g_build_model(params)
    local input = nn.Identity()()
    local target = nn.Identity()()
    local context = nn.Identity()()
    local time = nn.Identity()()
    local hid, shareList = build_memory(params, input, context, time)
    local z = nn.LinearNB(params.edim, params.nwords)(hid[#hid])
    local pred = nn.LogSoftMax()(z)                 --Apply softmax at last stage for prdiction
    local costl = nn.ClassNLLCriterion()
    costl.sizeAverage = false
    local cost = costl({pred, target})              --calculate accuracy
    local model = nn.gModule({input, target, context, time}, {cost})
    model:cuda()                                    --attach the cuda flag
    -- IMPORTANT! do weight sharing after model is in cuda
    for i = 1,#shareList do
        local m1 = shareList[i][1].data.module
        for j = 2,#shareList[i] do
            local m2 = shareList[i][j].data.module
            m2:share(m1,'weight','bias','gradWeight','gradBias')
        end
    end
    return model
end
