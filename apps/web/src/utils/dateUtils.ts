import { format, parseISO } from 'date-fns';

export function formatDate(date: Date | string, formatString: string = 'MMM d, yyyy'): string {
  if (typeof date === 'string') {
    try {
      return format(parseISO(date), formatString);
    } catch (error) {
      console.error('Error parsing date string:', error);
      return date;
    }
  }
  
  try {
    return format(date, formatString);
  } catch (error) {
    console.error('Error formatting date:', error);
    return date.toString();
  }
}

export function formatTimeAgo(date: Date | string): string {
  const now = new Date();
  const givenDate = typeof date === 'string' ? parseISO(date) : date;
  
  const seconds = Math.floor((now.getTime() - givenDate.getTime()) / 1000);
  
  if (seconds < 60) {
    return 'just now';
  }
  
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }
  
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }
  
  const days = Math.floor(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }
  
  return formatDate(givenDate, 'MMM d');
} 