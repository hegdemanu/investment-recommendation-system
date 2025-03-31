export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-8">
      <div className="max-w-5xl w-full">
        <h1 className="text-4xl font-bold mb-6 text-center">Investment Recommendation System</h1>
        
        <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200 mb-8">
          <h2 className="text-2xl font-semibold mb-4">Welcome to Your Dashboard</h2>
          <p className="mb-4">
            This platform provides personalized investment recommendations based on your risk profile and market analysis.
          </p>
          <button className="bg-blue-600 px-6 py-3 rounded-md font-medium text-white">
            Get Started
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md text-center border border-gray-100">
            <h3 className="text-xl font-semibold">Portfolio Analysis</h3>
            <p className="mt-2">Review your current investments</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md text-center border border-gray-100">
            <h3 className="text-xl font-semibold">Market Insights</h3>
            <p className="mt-2">Real-time market data</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md text-center border border-gray-100">
            <h3 className="text-xl font-semibold">Smart Recommendations</h3>
            <p className="mt-2">AI-powered investment suggestions</p>
          </div>
        </div>
        
        <div className="bg-blue-600 p-10 text-white text-center rounded-2xl shadow-lg">
          <h2 className="text-2xl font-bold mb-4">Ready to invest smarter?</h2>
          <p className="mb-6">Our AI-powered platform helps you make informed investment decisions.</p>
          <button className="bg-white text-blue-600 px-6 py-3 rounded-md font-medium">
            Learn More
          </button>
        </div>
      </div>
    </main>
  );
}
