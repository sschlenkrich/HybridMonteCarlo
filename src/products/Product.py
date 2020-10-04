#!/usr/bin/python

class Product:

    # We specify a common interface for our concrete products

    # return a list of payoffs V with payTime > obsTime
    # written as future present value observed at obsTime
    def cashFlows(self, obsTime):
        raise NotImplementedError('Implementation of method cashFlows required.')

    # return a time line of future cash flows per observation times
    def timeLine(self, obsTimes):
        return { t : self.cashFlows(t) for t in obsTimes }

# implement operations on time lines like join, observation times, ...