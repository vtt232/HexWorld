
function Base.findmax(f::Function, xs)
    f_max = -Inf
    x_max = first(xs)
    for x in xs
        v = f(x)
        if v > f_max
            f_max, x_max = v, x
        end
    end
    return f_max, x_max
end

Base.argmax(f::Function, xs) = findmax(f, xs)[2]

struct DiscountFactor
    y::Float64

    function DiscountFactor()
        new(0.9)
    end
end

struct StateSpace
    x::Vector{Int64}

    function StateSpace()
        x=collect(1:7)
        new(x)
        
    end
end 

struct ActionSpace
    x::Vector{String}

    function ActionSpace()
        x=["UpRight","Right","DownRight","UpLeft","Left","DownLeft"]
        new(x)
    end
end

function Transition(fromState,action,toState)
    if fromState==1
        if toState==2&&action=="UpLeft"
            return 1.0;
        else
            return 0.0;
        end
    elseif fromState==2
        if toState==3&&action=="UpLeft"
            return 1.0;
        elseif toState==4&&action=="UpRight"
            return 1.0;
        elseif toState==1&&action=="DownRight"
            return 1.0;
        else return 0.0;
        end
    elseif fromState==3
        if toState==4&&action=="Right"
            return 1.0;
        elseif toState==5 && action=="UpRight"
            return 1.0;
        elseif toState==2 && action=="DownRight"
            return 1.0;
        else return 0.0;
        end
    elseif fromState==4
        if toState==2 && action=="DownLeft"
            return 1.0
        elseif toState==5 && action=="UpLeft"
            return 1.0
        elseif toState==3 && action=="Left"
            return 1.0
        elseif toState==6 && action=="UpRight"
            return 1.0
        else return 0.0
        end
    elseif fromState==5
        if toState==3 && action=="DownLeft"
            return 1.0
        elseif toState==4&&action=="DownRight"
            return 1.0
        elseif toState==6&&action=="Right"
            return 1.0
        else return 0.0
        end
    elseif fromState==6
        if toState==5 && action=="Left"
            return 1.0
        elseif toState==4 && action=="DownLeft"
            return 1.0 
        elseif toState==7 && action=="Right"
            return 1.0
        else return 0.0
        end
    elseif fromState==7
        return 0.0
    else return 0.0
    end 
end

function Reward(fromState, action)
    if fromState==1
        if action=="UpRight"
            return 0;
        elseif action=="Right"
            return 0;
        elseif action=="DownRight"
            return 0;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    elseif fromState==2
        if action=="UpRight"
            return -2;
        elseif action=="Right"
            return 0;
        elseif action=="DownRight"
            return 0;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    elseif fromState==3
        if action=="UpRight"
            return 0;
        elseif action=="Right"
            return -2;
        elseif action=="DownRight"
            return 0;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    elseif fromState==4
        if action=="UpRight"
            return 0;
        elseif action=="Right"
            return 0;
        elseif action=="DownRight"
            return 0;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    elseif fromState==5
        if action=="UpRight"
            return 0;
        elseif action=="Right"
            return 0;
        elseif action=="DownRight"
            return -2;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    elseif fromState==6
        if action=="UpRight"
            return 0;
        elseif action=="Right"
            return 2;
        elseif action=="DownRight"
            return 0;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    elseif fromState==7
        if action=="UpRight"
            return 0;
        elseif action=="Right"
            return 0;
        elseif action=="DownRight"
            return 0;
        elseif action=="UpLeft"
            return 0;
        elseif action=="Left"
            return 0;
        elseif action=="DownLeft"
            return 0;
        end
    else return 0

    end
end


struct MDP
    γ # discount factor
    𝒮 # state space
    𝒜 # action space
    T # transition function
    R # reward function

    function MDP(discount,stateSpace,actionSpace,transitionFunction,rewardFunction)
        new(discount,stateSpace,actionSpace,transitionFunction,rewardFunction)
    end
end

function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end
    
function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end


struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MDP, U, s)
    u, a = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end
(π::ValueFunctionPolicy)(s) = greedy(π.𝒫, π.U, s).a

function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
end

struct ValueIteration
    k_max # maximum number of iterations
end

function solve(M::ValueIteration, 𝒫::MDP)
    U = [0.0 for s in 𝒫.𝒮]
    for k = 1:M.k_max
        U = [backup(𝒫, U, s) for s in 𝒫.𝒮]
    end
    return ValueFunctionPolicy(𝒫, U)
end

numbers=ValueIteration(2)
p=MDP(0.9,collect(1:7),["UpRight","Right","DownRight","UpLeft","Left","DownLeft"],Transition,Reward)
answer=solve(numbers,p)
answer(6)

