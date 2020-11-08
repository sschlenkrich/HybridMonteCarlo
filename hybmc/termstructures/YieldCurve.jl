
struct YieldCurve
    rate::Float64
end

function discount(self::YieldCurve, t::Float64)
    return exp(-self.rate*t)
end
