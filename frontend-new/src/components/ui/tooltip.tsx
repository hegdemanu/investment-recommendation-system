import * as React from "react";
import { cn } from "@/lib/utils";

interface TooltipProps {
  children: React.ReactNode;
  content: string;
  position?: "top" | "bottom" | "left" | "right";
  className?: string;
}

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  position = "top",
  className,
}) => {
  const [isVisible, setIsVisible] = React.useState(false);
  const [coords, setCoords] = React.useState({ x: 0, y: 0 });
  const tooltipRef = React.useRef<HTMLDivElement>(null);
  const childRef = React.useRef<HTMLDivElement>(null);

  const positionTooltip = React.useCallback(() => {
    if (!childRef.current || !tooltipRef.current) return;

    const childRect = childRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();

    const gap = 8; // Gap between tooltip and child element

    let x = 0;
    let y = 0;

    if (position === "top") {
      x = childRect.left + childRect.width / 2 - tooltipRect.width / 2;
      y = childRect.top - tooltipRect.height - gap;
    } else if (position === "bottom") {
      x = childRect.left + childRect.width / 2 - tooltipRect.width / 2;
      y = childRect.bottom + gap;
    } else if (position === "left") {
      x = childRect.left - tooltipRect.width - gap;
      y = childRect.top + childRect.height / 2 - tooltipRect.height / 2;
    } else if (position === "right") {
      x = childRect.right + gap;
      y = childRect.top + childRect.height / 2 - tooltipRect.height / 2;
    }

    // Adjust position to ensure tooltip stays within viewport
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight,
    };

    if (x < 0) x = 0;
    if (x + tooltipRect.width > viewport.width) {
      x = viewport.width - tooltipRect.width;
    }

    if (y < 0) y = 0;
    if (y + tooltipRect.height > viewport.height) {
      y = viewport.height - tooltipRect.height;
    }

    setCoords({ x, y });
  }, [position]);

  React.useEffect(() => {
    if (isVisible) {
      positionTooltip();
      window.addEventListener("resize", positionTooltip);
      window.addEventListener("scroll", positionTooltip);
    }

    return () => {
      window.removeEventListener("resize", positionTooltip);
      window.removeEventListener("scroll", positionTooltip);
    };
  }, [isVisible, positionTooltip]);

  return (
    <>
      <div
        ref={childRef}
        className="inline-block"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      >
        {children}
      </div>
      {isVisible && (
        <div
          ref={tooltipRef}
          className={cn(
            "fixed z-50 px-3 py-2 text-xs font-medium text-white bg-gray-800 dark:bg-gray-700 rounded-md shadow-md max-w-xs pointer-events-none transition-opacity duration-200",
            className
          )}
          style={{
            top: `${coords.y}px`,
            left: `${coords.x}px`,
            opacity: isVisible ? 1 : 0,
          }}
        >
          {content}
          <div
            className={cn(
              "absolute w-2 h-2 bg-gray-800 dark:bg-gray-700 transform rotate-45",
              position === "top" && "bottom-0 left-1/2 -translate-x-1/2 translate-y-1",
              position === "bottom" && "top-0 left-1/2 -translate-x-1/2 -translate-y-1",
              position === "left" && "right-0 top-1/2 translate-x-1 -translate-y-1/2",
              position === "right" && "left-0 top-1/2 -translate-x-1 -translate-y-1/2"
            )}
          />
        </div>
      )}
    </>
  );
}; 