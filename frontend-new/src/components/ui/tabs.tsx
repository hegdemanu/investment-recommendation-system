import * as React from "react";

interface TabsProps {
  children: React.ReactNode;
  defaultValue?: string;
  value?: string;
  onValueChange?: (value: string) => void;
}

interface TabsContextValue {
  value: string;
  onValueChange: (value: string) => void;
}

const TabsContext = React.createContext<TabsContextValue | undefined>(undefined);

const Tabs: React.FC<TabsProps> = ({
  children,
  defaultValue,
  value: controlledValue,
  onValueChange,
}) => {
  const [value, setValue] = React.useState(defaultValue || "");

  const handleValueChange = React.useCallback((newValue: string) => {
    setValue(newValue);
    onValueChange?.(newValue);
  }, [onValueChange]);

  const contextValue = React.useMemo(() => ({
    value: controlledValue !== undefined ? controlledValue : value,
    onValueChange: handleValueChange,
  }), [controlledValue, value, handleValueChange]);

  return (
    <TabsContext.Provider value={contextValue}>
      <div className="space-y-2">{children}</div>
    </TabsContext.Provider>
  );
};

interface TabsListProps {
  children: React.ReactNode;
}

const TabsList: React.FC<TabsListProps> = ({ children }) => {
  return (
    <div className="flex border-b border-gray-200 dark:border-gray-700">
      {children}
    </div>
  );
};

interface TabsTriggerProps {
  children: React.ReactNode;
  value: string;
}

const TabsTrigger: React.FC<TabsTriggerProps> = ({ children, value }) => {
  const context = React.useContext(TabsContext);

  if (!context) {
    throw new Error("TabsTrigger must be used within a Tabs component");
  }

  const { value: selectedValue, onValueChange } = context;
  const isSelected = selectedValue === value;

  return (
    <button
      className={`px-4 py-2 font-medium text-sm focus:outline-none ${
        isSelected
          ? "border-b-2 border-primary text-primary"
          : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
      }`}
      onClick={() => onValueChange(value)}
    >
      {children}
    </button>
  );
};

interface TabsContentProps {
  children: React.ReactNode;
  value: string;
}

const TabsContent: React.FC<TabsContentProps> = ({ children, value }) => {
  const context = React.useContext(TabsContext);

  if (!context) {
    throw new Error("TabsContent must be used within a Tabs component");
  }

  const { value: selectedValue } = context;

  if (selectedValue !== value) {
    return null;
  }

  return <div className="pt-4">{children}</div>;
};

export { Tabs, TabsList, TabsTrigger, TabsContent }; 