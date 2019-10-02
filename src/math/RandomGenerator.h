#include <string>
#include <vector>

class RandomGenerator{
	struct Impl;
	Impl* impl = nullptr;
public:
	static std::vector<std::string> supportList();
	static std::string usage();
	static bool isSupported(const std::string& name);

	void init(const std::vector<std::string>& param,
		const double offset = 0.0, const unsigned seed = 123456u);
	void update(const double off);
	void update(const std::vector<std::string>& param);
	double generate();
};
