import * as React from "react";

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  children: React.ReactNode;
  value?: string;
  onValueChange?: (value: string) => void;
}

const Select: React.FC<SelectProps> = ({
  children,
  value,
  onValueChange,
  onChange,
  ...props
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    if (onValueChange) {
      onValueChange(e.target.value);
    }
    onChange?.(e);
  };

  return (
    <select
      className="block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm placeholder-gray-400 shadow-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
      value={value}
      onChange={handleChange}
      {...props}
    >
      {children}
    </select>
  );
};

interface SelectTriggerProps {
  children: React.ReactNode;
  className?: string;
}

const SelectTrigger: React.FC<SelectTriggerProps> = ({
  children,
  className = "",
}) => {
  return (
    <div className={`relative ${className}`}>
      {children}
      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
        <svg
          className="h-4 w-4 fill-current"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
        >
          <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" />
        </svg>
      </div>
    </div>
  );
};

interface SelectValueProps {
  placeholder?: string;
}

const SelectValue: React.FC<SelectValueProps> = ({ placeholder }) => {
  return <span className="block truncate">{placeholder}</span>;
};

interface SelectContentProps {
  children: React.ReactNode;
}

const SelectContent: React.FC<SelectContentProps> = ({ children }) => {
  return (
    <div className="mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none dark:bg-gray-800 sm:text-sm">
      {children}
    </div>
  );
};

interface SelectItemProps extends React.HTMLAttributes<HTMLOptionElement> {
  children: React.ReactNode;
  value: string;
}

const SelectItem: React.FC<SelectItemProps> = ({
  children,
  value,
  ...props
}) => {
  return (
    <option value={value} {...props}>
      {children}
    </option>
  );
};

export { Select, SelectTrigger, SelectValue, SelectContent, SelectItem }; 