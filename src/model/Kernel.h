#pragma once
#include <vector>
#include <string>

class Kernel
{
public:
	virtual void init(const std::string& param) = 0;
	virtual bool checkData(const size_t nx, const size_t ny) = 0;
	virtual std::string name() const = 0;
	std::string parameter() const;
	virtual int lengthParameter() const = 0;
	virtual bool needInitParameterByData() const; // default false

	// default doing nothing
	virtual void initVariables(const std::vector<double>& x,
		std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph);
	virtual int lengthHidden() const; // default 0

	virtual std::vector<double> predict(const std::vector<double>& x, const std::vector<double>& w) const = 0;
	virtual int classify(const double p) const = 0;
	
	virtual double loss(const std::vector<double>& pred, const std::vector<double>& label) const = 0;
	virtual std::vector<double> gradient(const std::vector<double>& x,
		const std::vector<double>& w, const std::vector<double>& y, std::vector<double>* ph = nullptr) const = 0;

protected:
	std::string param;
	void initBasic(const std::string& param);
};
